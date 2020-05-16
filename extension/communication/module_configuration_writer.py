from pathlib import Path
from typing import NamedTuple, List
from os import fsync, rename

from .metric_to_id_mapping import InstrumentationFunction
from util.atomic_file import atomic_write

class ModuleConfiguration(NamedTuple):
    pids_to_instrument: List[int]
    instrumentation_functions: List[InstrumentationFunction] = [ InstrumentationFunction.INSTRUCTIONS_COUNT ]

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
