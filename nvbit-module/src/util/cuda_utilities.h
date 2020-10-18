#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

#define checkCudaErrors(val) util::checkError((val), #val, __FILE__, __LINE__)

namespace util {

void checkError(cudaError_t result, const char* calledFunc, std::string file, int line);
void checkError(CUresult result, const char* calledFunc, std::string file, int line);

} 