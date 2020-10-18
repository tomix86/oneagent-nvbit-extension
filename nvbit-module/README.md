# Native module for OneAgent NVBit extension

## Overview

Native module for gathering kernel execution performance metrics via code instrumentation.
It is compiled into a dynamic library, which then needs to be [preloaded](https://man7.org/linux/man-pages/man8/ld.so.8.html) into the process one wishes to monitor.

Example usage:

```sh
ONEAGENT_NVBIT_EXTENSION_CONF_FILE=<path-to>/nvbit-module.conf LD_PRELOAD=<path-to>/libnvbit-module.so <the application being instrumented>
```

## External dependencies

| Dependency                                                 | Tested version |
|------------------------------------------------------------|----------------|
|[spdlog](https://github.com/gabime/spdlog)                  | 1.3.1          |
|[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 11.0           |
|[NVBit](https://github.com/NVlabs/NVBit)                    | 1.5            |
|[Boost](https://www.boost.org/)                             | 1.71.0         |
|[Google Test](https://github.com/google/googletest)         | 1.10.0         |
|[CMake](https://cmake.org/download/)                        | 3.18.4         |
|[vcpkg](https://github.com/Microsoft/vcpkg)                 | N/A            |
|Compiler with C++20 support (C++17 for CUDA)                | gcc 10.2.0     |

## Building

### Set up vcpkg

```sh
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install spdlog:x64-linux boost-program-options:x64-linux gtest:x64-linux
```

### Download NVBit

NVBit does not require separate compilation as documented in README ("_Getting Started with NVBit_" section) located in the root directory of NVBit release package.

### Build the project

```sh
mkdir build
cd build
cmake -G "Unix Makefiles" -DNVBIT_PATH="<path_to_nvbit_release>" -DCMAKE_TOOLCHAIN_FILE="<vcpkg_directory>/scripts/buildsystems/vcpkg.cmake" ..
```

## Configuration

The module is configured twofold:

1. startup configuration is read once during start from the file specified via `ONEAGENT_NVBIT_EXTENSION_CONF_FILE` environment variable,
2. runtime configuration is read every `runtime_config_polling_interval` seconds from file specified via `runtime_config_path`.

### Startup configuration

Startup configuration needs to be provided upfront via `ONEAGENT_NVBIT_EXTENSION_CONF_FILE` environment variable.
Lintes starting with `#` are treated as comments and ignored.
The list of settings is as denoted in the table below.

| Key                               | Value type            | Default value | Description                                       |
|-----------------------------------|-----------------------|---------------|---------------------------------------------------|
| `logfile`                         | Valid filesystem path | _unset_       | Path to log file                                  |
| `runtime_config_path`             | Valid filesystem path | _unset_       | Path to runtime configuration file                |
| `runtime_config_polling_interval` | Positive integer      | 10            | Runtime configuration polling internal in seconds |
| `measurements_output_dir`         | Valid filesystem path | _unset_       | Directory where measurements will be written to   |
| `verbose`                         | Boolean               | false         | Enable verbose (debug) logging                    |
| `count_warp_level`                | Boolean               | true          | Count warp level or thread level instructions     |
| `exclude_pred_off`                | Boolean               | false         | Exclude predicated off instruction from count     |
| `mangled_names`                   | Boolean               | true          | Print kernel names mangled or not                 |

See [nvbit-module.conf](res/nvbit-module.conf) for an example.

### Runtime configuration

Runtime configuration is created on the fly by Python extension and contains a list of pids that should be instrumented, along with instrumentation primitives to apply to each of them.

See [nvbit-module-runtime.conf](res/nvbit-module-runtime.conf) for an example.
For a detailed documentation of communication protocol, [see here](../docs/communication_endpoint.md).
