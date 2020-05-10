from pathlib import Path
from typing import NamedTuple, List
from os import fsync, rename

class InstrumentationFunctions:
    INSTRUCTIONS_COUNT: int = 0


class ModuleConfiguration(NamedTuple):
    pids_to_instrument: List[int]
    instrumentation_functions: List[int] = [ InstrumentationFunctions.INSTRUCTIONS_COUNT ]

class ModuleConfigurationWriter:
    confFilePath: Path = "/var/lib/dynatrace/oneagent/agent/runtime/nvbit-module-runtime.conf"
    instrumentation_enabled: bool = False

    def __init__(self, instrumentation_enabled: bool):
        self.instrumentation_enabled = instrumentation_enabled
    
    def write(self, config: ModuleConfiguration):
        #TODO: add a helper for atomic write
        tmpFilePath = self.confFilePath + ".tmp"
        with open(tmpFilePath, mode="w") as confFile:
            if self.instrumentation_enabled:                
                for pid in config.pids_to_instrument:
                    instrument_with = ','.join(str(id) for id in config.instrumentation_functions)
                    confFile.write(f"{pid}:{instrument_with}\n")
            confFile.flush()
            fsync(confFile.fileno())
        rename(tmpFilePath, self.confFilePath)