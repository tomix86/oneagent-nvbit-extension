from pynvml import *

def init():
    nvmlInit()
    driverVersion = nvmlSystemGetDriverVersion().decode('UTF-8')
    nvmlVersion = nvmlSystemGetNVMLVersion().decode('UTF-8')
    print(f"Driver version: {driverVersion}\nNVML version: {nvmlVersion}")


def shutdown():
    nvmlShutdown()

def enumDevices():
    deviceCount = nvmlDeviceGetCount()
    devices = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)

        deviceName = nvmlDeviceGetName(handle).decode('UTF-8')
        devices.append(f"Device {i}: {deviceName}")

    return devices