from typing import List

from pynvml import c_nvmlDevice_t, c_uint

SAMPLES_COUNT: int = 5
SAMPLING_INTERVAL: int = 2

Pid = c_uint
GPUProcesses = List[Pid]
DeviceHandle = c_nvmlDevice_t
