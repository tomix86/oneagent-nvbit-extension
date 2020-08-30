# Native module for OneAgent NVBit extension

## What is it

...

## External dependencies

* [spdlog](https://github.com/gabime/spdlog) (tested with v1.3.1)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with v11.0)
* [Boost](https://www.boost.org/) (tested with v1.71.0)
* [CMake](https://cmake.org/download/) (tested with v3.15.5)
* [vcpkg](https://github.com/Microsoft/vcpkg)
* Compiler with C++17 support (tested with gcc 9.3.0)

## Building

### Set up vcpkg

```sh
$ git clone https://github.com/Microsoft/vcpkg.git
$ cd vcpkg
$ ./bootstrap-vcpkg.sh
$ ./vcpkg integrate install
$ ./vcpkg install spdlog:x64-linux boost-program-options:x64-linux
```

### Build the project

```sh
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles" -DNVBIT_PATH="<path_to_nvbit_release>" -DCMAKE_TOOLCHAIN_FILE="<vcpkg_directory>/scripts/buildsystems/vcpkg.cmake" ..
```
