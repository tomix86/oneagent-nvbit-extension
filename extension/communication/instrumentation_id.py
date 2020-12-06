from enum import IntEnum
from typing import List

class InstrumentationId(IntEnum):
    INSTRUCTIONS_COUNT = 0
    OCCUPANCY = 1
    GMEM_ACCESS_COALESCENCE = 2

    @classmethod
    def get_metric_name(cls, id: int) -> str:
        if(id == cls.INSTRUCTIONS_COUNT):
            return "instructions_per_second"
        elif (id == cls.OCCUPANCY):
            return "gpu_occupancy"
        elif (id == cls.GMEM_ACCESS_COALESCENCE):
            return "gmem_access_coalescence"
        else:
            raise ValueError(f"Unknown metric identifier: {id}")

    @classmethod
    def get_metric_id(cls, name: str) -> str:
        if(name == "instructions_per_second"):
            return cls.INSTRUCTIONS_COUNT
        elif (name == "gpu_occupancy"):
            return cls.OCCUPANCY
        elif (name == "gmem_access_coalescence"):
            return cls.GMEM_ACCESS_COALESCENCE
        else:
            raise ValueError(f"Unknown metric name: {name}")

    @classmethod
    def aggregate_samples(cls, id: int, samples: List[float]) -> float:
        averaged_metrics = [ cls.OCCUPANCY, cls.GMEM_ACCESS_COALESCENCE ]
        per_second_metrics = [ cls.INSTRUCTIONS_COUNT ]
        
        if(id in averaged_metrics):
            return sum(samples) / len(samples)
        elif(id in per_second_metrics):
            return sum(samples) / 60
        else:
            return sum(samples)
