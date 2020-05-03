from datetime import datetime

from utilities.utilities import nvml_error_to_string
from nvml_bridge import nvml_bridge
from communication import request_handler


"""
For documentation see README.md
"""


if __name__ == "__main__":
    try:
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        nvml_bridge.init()
        request_handler.start(nvml_bridge.enumDevices)

    #    request_handler.stop()
        nvml_bridge.shutdown()
    except NVMLError as error:
        print(nvml_error_to_string(error))