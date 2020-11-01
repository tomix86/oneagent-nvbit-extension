from pathlib import Path
from typing import Dict, List, Tuple, Optional
from os import listdir, unlink, mkdir, umask

from .instrumentation_id import InstrumentationId

Metric = Tuple[int, float]
Measurements = Dict[int, List[Metric]]


class MeasurementsReader:
    __measurementsPath: Path = Path(
        "/var/lib/dynatrace/oneagent/agent/runtime/nvbit_module_measurements")

    @classmethod
    def createMeasurementsDir(cls) -> None:
        try:
            umask(0o000)
            mkdir(cls.__measurementsPath, 0o1777)
        except FileExistsError:
            pass
        finally:
            umask(0o022)

    @classmethod
    def __addMeasurement(cls, measurements: Measurements, pid: int, entry: Metric) -> None:
        try:
            measurements[pid].append(entry)
        except KeyError:
            measurements[pid] = [entry]

    @classmethod
    def __extractPid(cls, filename: str) -> Optional[int]:
        if filename.count("-") != 2:
            return None

        pid = filename.split("-")[0]
        if not pid.isdigit():
            return None

        return int(pid)

    @classmethod
    def read(cls) -> Measurements:
        measurements = {}
        for filename in listdir(cls.__measurementsPath):
            pid = cls.__extractPid(filename)
            if not pid:
                continue

            with open(cls.__measurementsPath / filename, mode="r") as measurementsFile:
                for line in measurementsFile.readlines():
                    if line.count(":") != 1:
                        continue

                    try:
                        metricId, measurement = line.split(":")
                        entry = (int(metricId), float(measurement))
                        cls.__addMeasurement(measurements, pid, entry)
                    except ValueError:
                        continue

            unlink(cls.__measurementsPath / filename)

        return measurements
