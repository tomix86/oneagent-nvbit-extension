from time import sleep
from typing import Dict, Tuple, List, Any

from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetGraphicsRunningProcesses, NVMLError, \
                    nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlDeviceGetName, nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion, nvmlSystemGetNVMLVersion
from ruxit.api.base_plugin import BasePlugin
from ruxit.api.data import PluginMeasurement
from ruxit.api.selectors import ExplicitPgiSelector
from ruxit.api.exceptions import ConfigException

from util.constants import DeviceHandle, GPUProcesses, Pid, SAMPLES_COUNT, SAMPLING_INTERVAL
from util.utilities import nvml_error_to_string

"""
For documentation see README.md
"""


class NVBitExtension(BasePlugin):
    devices_count: int = 0
    enable_debug_log: bool = False

    def raise_nvml_error(self, error: NVMLError) -> None:
        self.logger.warning(nvml_error_to_string(error))
        raise ConfigException(f"unexpected NVML error: {str(error)}") from error

    def log_debug(self, message: str) -> None:
        if self.enable_debug_log:
            self.logger.info("[NVBIT DEBUG]: " + message)

    def set_pgi_results(self, pgi_id: int, occupancy: int) -> None:
        if occupancy is not None:
            measurement = PluginMeasurement(key="gpu_occupancy", value=occupancy, entity_selector=ExplicitPgiSelector(pgi_id))
            self.results_builder.add_absolute_result(measurement)
        else:  # Note: if we don't send these metrics it won't appear on the WebUI, this is expected (otherwise we would display a timeseries that does not make any sense)
            self.log_debug(f"Skipping gpu_mem_used_by_pgi metric for PGIID={pgi_id:02x} as the occupancy reading is empty")

    def sample_processes_information(self, handle: DeviceHandle) -> GPUProcesses:
        try:
            # List processes with a compute context (e.g. CUDA applications)
            compute_processes = nvmlDeviceGetComputeRunningProcesses(handle)
            # List processes with a graphics context (eg. applications using OpenGL, DirectX)
            graphics_processes = nvmlDeviceGetGraphicsRunningProcesses(handle)
            # Note: a single process may have both the graphics and compute context active at the same time
        except NVMLError as err:
            self.logger.warning(nvml_error_to_string(err))
            return []

        processes = []
        for p in compute_processes + graphics_processes:
            processes.append(p.pid)

        self.log_debug(f"Sampled processes ({len(processes)}): {processes}")
        return processes

    def aggregate_data_from_multiple_devices(self, data: List[GPUProcesses]) -> GPUProcesses:
        gpu_processes = []
        for device_info in data:
            self.log_debug(f"Aggregating device data: {device_info}")
            gpu_processes += device_info

        gpu_processes = list(dict.fromkeys(gpu_processes))
        self.log_debug(f"Device data processes count: {len(gpu_processes)}")
        return gpu_processes

    def sample_devices_information(self) -> List[GPUProcesses]:
        data_for_devices = []
        for idx in range(self.devices_count):
            self.log_debug(f"Sampling GPU #{idx}")
            handle = nvmlDeviceGetHandleByIndex(idx)
            processes_info = self.sample_processes_information(handle)
            data_for_devices.append(processes_info)

        return data_for_devices

    def get_gpus_info(self) -> List[GPUProcesses]:
        # Gather first sample
        data_for_devices = self.sample_devices_information()

        for _ in range(1, SAMPLES_COUNT):
            new_sample = self.sample_devices_information()
            for idx in range(0, len(data_for_devices)):
                previous = data_for_devices[idx]
                current = new_sample[idx]
                # We're only interested in processes that appear in all the samples
                processes_info = [k for k in previous if k in current]
                data_for_devices[idx] = processes_info

            sleep(SAMPLING_INTERVAL)

        for device_data in data_for_devices:
            self.log_debug(f"Device info:")
            self.log_debug(f"...Number of processes using the GPU: {len(device_data)}")
            self.log_debug(f"...PIDs and memory usage of processes using the GPU: {device_data}")
        return data_for_devices

    def get_monitored_pgis_list(self, gpu_processes: GPUProcesses) -> Dict[int, object]:
        monitored_pgis = []

        pgi_list = self.find_all_processes(lambda process: process.pid in gpu_processes)
        for entry in pgi_list:
            pgi = entry[0]
            pid = entry[1].pid
            name = entry[1].process_name
            self.log_debug(f"{name} (pid: {pid}) from {pgi.group_name} process group"
                           f"(PGIID={pgi.group_instance_id:02x}, type={pgi.process_type}) is using the GPU")
            monitored_pgis.append(pgi)

        return { pgi.group_instance_id: pgi for pgi in monitored_pgis }

    def generate_metrics_for_pgis(self, gpu_processes_mem_usage: GPUProcesses, monitored_pgis: Dict) -> None:
        for pgi in monitored_pgis.values():
            self.log_debug(f"Processing '{pgi.group_name}' process group...")
            pgi_id = pgi.group_instance_id
            self.logger.info(f"Sending occupancy metric for '{pgi.group_name}' process group (PGIID={pgi_id:02x}, type={pgi.process_type})")
            self.set_pgi_results(pgi_id, 65)

    def detect_devices(self) -> None:
        self.devices_count = nvmlDeviceGetCount()
        for i in range(self.devices_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            device_name = nvmlDeviceGetName(handle).decode("UTF-8")
            self.logger.info(f"Device nr. {i}: '{device_name}'")

    def initialize(self, **kwargs) -> None:
        try:
            nvmlInit()
            driver_version = nvmlSystemGetDriverVersion().decode("UTF-8")
            nvml_version = nvmlSystemGetNVMLVersion().decode("UTF-8")
            self.logger.info(f"NVML initialized, driver version: {driver_version}, NVML version: {nvml_version}")
            self.detect_devices()
        except NVMLError as error:
            self.raise_nvml_error(error)

    def close(self, **kwargs) -> None:
        try:
            nvmlShutdown()
            self.logger.info(f"NVML shut down")
        except NVMLError as error:
            self.raise_nvml_error(error)

    def query(self, **kwargs) -> None:
        config = kwargs["config"]
        self.enable_debug_log = config["enable_debug_log"]

        try:
            data_for_devices = self.get_gpus_info()
            processes_running_on_gpu = self.aggregate_data_from_multiple_devices(data_for_devices)
            monitored_pgis = self.get_monitored_pgis_list(processes_running_on_gpu)
            self.generate_metrics_for_pgis(processes_running_on_gpu, monitored_pgis)
        except NVMLError as error:
            self.raise_nvml_error(error)
