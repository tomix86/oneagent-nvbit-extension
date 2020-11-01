from enum import IntEnum
from typing import List

class InstrumentationId(IntEnum):
    INSTRUCTIONS_COUNT = 0
    OCCUPANCY = 1

    @classmethod
    def get_metric_name(cls, id: int) -> str:
        if(id == cls.INSTRUCTIONS_COUNT):
            return "instructions_per_second"
        elif (id == cls.OCCUPANCY):
            return "gpu_occupancy"
        else:
            raise ValueError("Unknown metric identifier")

    @classmethod
    def aggregate_samples(cls, id: int, samples: List[float]) -> float:
        averaged_metrics = [ cls.OCCUPANCY ]
        per_second_metrics = [ cls.INSTRUCTIONS_COUNT ]
        
        if(id in averaged_metrics):
            return sum(samples) / len(samples)
        elif(id in per_second_metrics):
            return sum(samples) / 60
        else:
            return sum(samples)
