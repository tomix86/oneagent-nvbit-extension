from typing import Dict, List

from ruxit.api.base_plugin import BasePlugin
from ruxit.api.data import PluginMeasurement
from ruxit.api.selectors import ExplicitPgiSelector
from ruxit.api.exceptions import ConfigException

from communication.module_configuration_writer import ModuleConfigurationWriter, ModuleConfiguration

"""
For documentation see README.md
"""


class NVBitExtension(BasePlugin):
    devices_count: int = 0
    enable_debug_log: bool = False
    instrumentation_enabled: bool = True

    def log_debug(self, message: str) -> None:
        if self.enable_debug_log:
            self.logger.info("[NVBIT DEBUG]: " + message)

    def set_pgi_results(self, pgi_id: int, occupancy: int) -> None:
        if occupancy is not None:
            measurement = PluginMeasurement(key="gpu_occupancy", value=occupancy, entity_selector=ExplicitPgiSelector(pgi_id))
            self.results_builder.add_absolute_result(measurement)
        else:  # Note: if we don't send these metrics it won't appear on the WebUI, this is expected (otherwise we would display a timeseries that does not make any sense)
            self.log_debug(f"Skipping gpu_mem_used_by_pgi metric for PGIID={pgi_id:02x} as the occupancy reading is empty")

    def get_monitored_pgis_list(self, monitored_pg_names: List[str]) -> Dict[int, object]:
        monitored_pgis = []

        pgi_list = self.find_all_processes(lambda process: process.process_name in monitored_pg_names)
        for entry in pgi_list:
            pgi = entry[0]
            pid = entry[1].pid
            name = entry[1].process_name
            self.log_debug(f"Detected native NVBit module injection in {name} (pid: {pid}) from {pgi.group_name} process group"
                           f"(PGIID={pgi.group_instance_id:02x}, type={pgi.process_type})")
            monitored_pgis.append(pgi)

        return { pgi.group_instance_id: pgi for pgi in monitored_pgis }

    def generate_metrics_for_pgis(self, monitored_pgis: Dict) -> None:
        for pgi in monitored_pgis.values():
            self.log_debug(f"Processing '{pgi.group_name}' process group...")
            pgi_id = pgi.group_instance_id
            self.logger.info(f"Sending occupancy metric for '{pgi.group_name}' process group (PGIID={pgi_id:02x}, type={pgi.process_type})")
            self.set_pgi_results(pgi_id, 65)

    def initialize(self, **kwargs) -> None:
        self.logger.info(f"NVBit plugin initialized")

    def close(self, **kwargs) -> None:
        self.logger.info(f"NVBit plugin shut down")


    def query(self, **kwargs) -> None:
        config = kwargs["config"]
        self.enable_debug_log = config["enable_debug_log"]
        self.instrumentation_enabled = config["enable_intrumentation"]

        monitored_pg_names = config["monitored_pg_names"].split(",")
        monitored_pgis = self.get_monitored_pgis_list(monitored_pg_names)

        all_monitored_processes = []
        for pgi in monitored_pgis.values():
            all_monitored_processes += pgi.processes

        moduleConfiguration = ModuleConfiguration(pids_to_instrument = [process.pid for process in all_monitored_processes])
        self.log_debug(f"Instrumentation enabled {self.instrumentation_enabled }, configuration: {moduleConfiguration}")
        ModuleConfigurationWriter(self.instrumentation_enabled).write(moduleConfiguration)

        self.generate_metrics_for_pgis(monitored_pgis)
