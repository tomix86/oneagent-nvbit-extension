from enum import IntEnum

class InstrumentationFunction(IntEnum):
    INSTRUCTIONS_COUNT = 0

    @classmethod
    def get_metric_name(cls, id: int) -> str:
        if(id == cls.INSTRUCTIONS_COUNT):
            return "instructions_per_second"
        raise ValueError("Unknown metric identifier")