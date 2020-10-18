#include "cuda_utilities.h"

#include "Logger.h"

namespace util {

void checkError(cudaError_t result, const char* calledFunc, std::string file, int line) {
    if (!result) { return; }
    const auto relativeFilePath = file.substr(file.rfind("src/"));
    logging::info("{} failed ({}:{}) code {} ({})", calledFunc, relativeFilePath, line, result, cudaGetErrorString(result));
}

}