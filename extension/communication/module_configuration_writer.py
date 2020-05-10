from pathlib import Path
from typing import NamedTuple, List
from os import fsync, rename

from .metric_to_id_mapping import InstrumentationFunction

class ModuleConfiguration(NamedTuple):
    pids_to_instrument: List[int]
    instrumentation_functions: List[int] = [ InstrumentationFunction.INSTRUCTIONS_COUNT ]

class ModuleConfigurationWriter:
    __confFilePath: Path = "/var/lib/dynatrace/oneagent/agent/runtime/nvbit-module-runtime.conf"
    __instrumentation_enabled: bool = False

    def __init__(self, instrumentation_enabled: bool):
        self.__instrumentation_enabled = instrumentation_enabled
    
    def write(self, config: ModuleConfiguration) -> None:
        #TODO: add a helper for atomic write
        tmpFilePath = self.__confFilePath + ".tmp"
        with open(tmpFilePath, mode="w") as confFile:
            if self.__instrumentation_enabled:                
                for pid in config.pids_to_instrument:
                    instrument_with = ','.join(str(id) for id in config.instrumentation_functions)
                    confFile.write(f"{pid}:{instrument_with}\n")
            confFile.flush()
            fsync(confFile.fileno())
        rename(tmpFilePath, self.__confFilePath)