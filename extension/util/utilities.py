from pynvml import NVMLError, NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_NO_PERMISSION


def nvml_error_to_string(error: NVMLError) -> str:
    if error.value == NVML_ERROR_NOT_SUPPORTED:
        return "N/A"
    if error.value == NVML_ERROR_NO_PERMISSION:
        return "Access denied"
    else:
        return str(error)