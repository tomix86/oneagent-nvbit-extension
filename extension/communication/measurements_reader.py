from pathlib import Path
from typing import Dict, List, Tuple
from os import listdir, unlink, mkdir, umask

from .metric_to_id_mapping import InstrumentationFunction

class MeasurementsReader:
    __measurementsPath: Path = Path("/var/lib/dynatrace/oneagent/agent/runtime/nvbit_module_measurements")

    @classmethod
    def createMeasurementsDir(self) -> None:
        try:
            umask(0o000)
            mkdir(self.__measurementsPath, 0o1777)
        except FileExistsError:
            pass
        finally:
            umask(0o022)

    @classmethod
    def read(self) -> Dict[int, List[Tuple[int, float]]]:
        measurements = {}
        for filePath in listdir(self.__measurementsPath):
            #TODO: error checking for file name
            pid = int(filePath.split("-")[0])
            with open(self.__measurementsPath / filePath, mode="r") as measurementsFile:
                for line in measurementsFile.readlines():
                    #TODO: error checking for line syntax
                    metricId, measurement = line.split(":")
                    entry = (int(metricId), float(measurement))
                    try:
                        measurements[pid].append(entry)
                    except KeyError:
                        measurements[pid] = [entry]
            unlink(self.__measurementsPath / filePath)
        
        return measurements