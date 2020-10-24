cmake_minimum_required(VERSION 3.18.4)
project(nvbit-module CUDA CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(spdlog REQUIRED)
find_package(Boost REQUIRED program_options)
find_package(GTest REQUIRED)

set(NVBIT_PATH "" CACHE FILEPATH "Path to NVBit release")
if(NOT EXISTS "${NVBIT_PATH}/core/nvbit_tool.h")
  message(SEND_ERROR "Couldn't find nvbit_tool.h")
endif()
include_directories("${NVBIT_PATH}/core")
link_directories("${NVBIT_PATH}/core")

set(SRC "${CMAKE_SOURCE_DIR}/src")

include_directories("${SRC}")

file(GLOB_RECURSE SOURCES "${SRC}/*.cpp")
list(REMOVE_ITEM SOURCES "${SRC}/main.cpp")

file(GLOB_RECURSE INJECTEE_SOURCES "${SRC}/device_functions/*.injectee.cu")
file(GLOB_RECURSE CUDA_SOURCES "${SRC}/*.cu")
list(REMOVE_ITEM CUDA_SOURCES ${INJECTEE_SOURCES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES --compiler-options=-Wall,-fPIC,-Wextra,-Werror")
set(CMAKE_CUDA_ARCHITECTURES 52)
set(PEDANTIC_ERRORS -Wall -Wextra -pedantic -Werror)

add_library(entrypoint OBJECT ${CUDA_SOURCES})
target_compile_options(entrypoint PRIVATE -O3 --relocatable-device-code=true --ptxas-options=-cloning=no)
set_target_properties(entrypoint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(entrypoint PRIVATE spdlog::spdlog)

add_library(injected_functions OBJECT ${INJECTEE_SOURCES})
target_compile_options(injected_functions PRIVATE --maxrregcount=24 --keep-device-functions --ptxas-options=--compile-as-tools-patch)

add_library(nvbit-module SHARED ${SOURCES} "${SRC}/main.cpp")
target_compile_options(nvbit-module PRIVATE ${PEDANTIC_ERRORS})
set_target_properties(nvbit-module PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nvbit-module PRIVATE
                      spdlog::spdlog ${Boost_LIBRARIES} nvbit cuda
                      entrypoint injected_functions)



# Unit tests configuration
include(GoogleTest)
include(CTest)
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")
file(GLOB_RECURSE CUDA_TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cu")

add_library(cuda_tests OBJECT ${CUDA_TEST_SOURCES})
target_compile_options(cuda_tests PRIVATE -O3)
target_link_libraries(cuda_tests PRIVATE GTest::GTest GTest::Main ${Boost_LIBRARIES} cuda)                

add_executable(nvbit-module-test ${TEST_SOURCES} ${SOURCES})
target_compile_options(nvbit-module-test PRIVATE ${PEDANTIC_ERRORS})
target_link_libraries(nvbit-module-test PRIVATE
                      GTest::GTest GTest::Main                      
                      spdlog::spdlog ${Boost_LIBRARIES}
                      cuda_tests)

gtest_discover_tests(nvbit-module-test)