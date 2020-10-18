cmake_minimum_required(VERSION 3.15)
project(nvbit-module CUDA CXX C)

find_package(spdlog REQUIRED)
find_package(Boost REQUIRED program_options)
find_package(GTest REQUIRED)

set(NVBIT_PATH "" CACHE FILEPATH "Path to NVBit release")
if(NOT EXISTS "${NVBIT_PATH}/core/nvbit_tool.h")
  message(SEND_ERROR "Couldn't find nvbit_tool.h")
endif()
include_directories("${NVBIT_PATH}/core")
link_directories("${NVBIT_PATH}/core")

include_directories("${CMAKE_SOURCE_DIR}/src")

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

file(GLOB_RECURSE DEVICE_SOURCES "${CMAKE_SOURCE_DIR}/src/device_functions/*.injectee.cu")
file(GLOB_RECURSE ENTRYPOINT_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
list(REMOVE_ITEM ENTRYPOINT_SOURCES ${DEVICE_SOURCES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_52 -D_FORCE_INLINES --compiler-options=-Wall,-fPIC")

add_library(entrypoint OBJECT ${ENTRYPOINT_SOURCES})
target_compile_options(entrypoint PRIVATE -O3 --relocatable-device-code=true --ptxas-options=-cloning=no)
target_link_libraries(entrypoint PRIVATE spdlog::spdlog)

add_library(injected_functions OBJECT ${DEVICE_SOURCES})
target_compile_options(injected_functions PRIVATE --maxrregcount=24 --keep-device-functions --ptxas-options=--compile-as-tools-patch)

add_library(nvbit-module SHARED ${SOURCES})
set_target_properties(nvbit-module PROPERTIES CXX_STANDARD 17)
set_target_properties(nvbit-module PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nvbit-module PRIVATE
                      spdlog::spdlog ${Boost_LIBRARIES} nvbit cuda
                      entrypoint injected_functions)


# Unit tests configuration
include(GoogleTest)
include(CTest)
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp" "${CMAKE_SOURCE_DIR}/test/*.cu")

add_executable(nvbit-module-test ${TEST_SOURCES})
target_link_libraries(nvbit-module-test PRIVATE GTest::GTest GTest::Main)

gtest_discover_tests(nvbit-module-test)