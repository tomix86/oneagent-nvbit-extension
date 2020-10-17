#pragma once

#include <cuda_runtime.h>
#include <string>

#define checkCudaErrors(val) checkError((val), #val, __FILE__, __LINE__)
void checkError(cudaError_t result, const char* calledFunc, std::string file, int line);
