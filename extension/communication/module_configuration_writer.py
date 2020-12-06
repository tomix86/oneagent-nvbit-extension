from pathlib import Path
from typing import NamedTuple, List, Dict, Any
from os import fsync, rename

from .instrumentation_id import InstrumentationId
from util.atomic_file import atomic_write

def get_intrumentation_ids(config: Dict[str, Any]) -> List[InstrumentationId]:
    intrumentation_ids = []
    if config["instrumentation_occupancy"]:
        intrumentation_ids.append(InstrumentationId.OCCUPANCY)

    if config["instrumentation_code_injection"] != "none":
        intrumentation_ids.append(InstrumentationId.get_metric_id(config["instrumentation_code_injection"]))
    
    return intrumentation_ids

class ModuleConfiguration(NamedTuple):
    pids_to_instrument: List[int]
    instrumentation_functions: List[InstrumentationId]

class ModuleConfigurationWriter:
    __confFilePath: Path = "/var/lib/dynatrace/oneagent/agent/runtime/nvbit-module-runtime.conf"
    __instrumentation_enabled: bool = False

    def __init__(self, instrumentation_enabled: bool):
        self.__instrumentation_enabled = instrumentation_enabled
    
    def write(self, config: ModuleConfiguration) -> None:
        with atomic_write(self.__confFilePath) as confFile:
            if not self.__instrumentation_enabled:
                return
            for pid in config.pids_to_instrument:
                instrument_with = ','.join(str(id.value) for id in config.instrumentation_functions)
                confFile.write(f"{pid}:{instrument_with}\n")
