from enum import IntEnum

#TODO: use enum here
class InstrumentationFunction:
    INSTRUCTIONS_COUNT: int = 0

    @classmethod
    def get_metric_name(self, id: int) -> str:
        if(id == self.INSTRUCTIONS_COUNT):
            return "instructions_per_second"
        raise "Unknown metric identifier"