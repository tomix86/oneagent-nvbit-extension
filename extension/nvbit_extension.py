from typing import Dict, List, Tuple

from ruxit.api.base_plugin import BasePlugin
from ruxit.api.data import PluginMeasurement
from ruxit.api.selectors import ExplicitPgiSelector
from ruxit.api.exceptions import ConfigException

from communication.module_configuration_writer import ModuleConfigurationWriter, ModuleConfiguration
from communication.measurements_reader import MeasurementsReader, Measurements
from communication.metric_to_id_mapping import InstrumentationFunction

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

    def set_pgi_results(self, pgi_id: int, key:str, value: int) -> None:
        if value is not None:
            measurement = PluginMeasurement(key=key, value=value, entity_selector=ExplicitPgiSelector(pgi_id))
            self.results_builder.add_absolute_result(measurement)
        else:  # Note: if we don't send these metrics it won't appear on the WebUI, this is expected (otherwise we would display a timeseries that does not make any sense)
            self.log_debug(f"Skipping {key} metric for PGIID={pgi_id:02x} as the reading is empty")

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

    def generate_metrics_for_pgis(self, monitored_pgis: Dict, metrics: Dict[int, Dict[str, float]]) -> None:
        for pgi in monitored_pgis.values():
            self.log_debug(f"Processing '{pgi.group_name}' process group...")
            pgi_metrics = {}
            for process in pgi.processes:
                if process.pid not in metrics:
                    continue
                
                #TODO: aggregation over multiple processes (merge dicts)
                pgi_metrics = metrics[process.pid]

            pgi_id = pgi.group_instance_id
            for key, value in pgi_metrics.items():
                self.logger.info(f"Sending '{key} = {value}' metric for '{pgi.group_name}' process group (PGIID={pgi_id:02x}, type={pgi.process_type})")
                self.set_pgi_results(pgi_id, key, value)

    def addMetricValue(self, metrics: Dict[int, Dict[str, float]], name: str, value: float):
        try:
            metrics[name] += value
        except KeyError:
            metrics[name] = value

    def process_measurements(self, measurements: Measurements) -> Dict[int, Dict[str, float]]:
        metrics = {}
        for pid, raw_metrics in measurements.items():
            aggregated_pid_metrics = {}
            for id, value in raw_metrics:
                name = InstrumentationFunction.get_metric_name(id)
                self.addMetricValue(aggregated_pid_metrics, name, value)
            
            metrics[pid] = aggregated_pid_metrics

        #TODO: metrics postprocessing, e.g. divide INSTRUCTIONS_COUNT by 60

        return metrics
                

    def initialize(self, **kwargs) -> None:
        self.logger.info(f"NVBit plugin initialized")
        MeasurementsReader.createMeasurementsDir()

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

        measurements = MeasurementsReader.read()
        self.log_debug(f"Measurements: {measurements}")

        metrics = self.process_measurements(measurements)
        self.log_debug(f"Aggregated metrics: {metrics}")

        self.generate_metrics_for_pgis(monitored_pgis, metrics)
