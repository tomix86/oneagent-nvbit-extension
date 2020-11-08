#include "Logger.h"
#include "cuda_utilities.h"

#include <fmt/core.h>

namespace util {

void checkError(cudaError_t result, const char* calledFunc, std::string file, int line) {
	if (result == cudaSuccess) {
		return;
	}

	const auto relativeFilePath{file.substr(file.rfind("src/"))};
	logging::warning("{} failed ({}:{}) code {} ({})", calledFunc, relativeFilePath, line, result, cudaGetErrorString(result));
}

void checkError(CUresult result, const char* calledFunc, std::string file, int line) {
	if (result == CUDA_SUCCESS) {
		return;
	}

	const auto relativeFilePath{file.substr(file.rfind("src/"))};

	const char* errorString{};
	cuGetErrorString(result, &errorString);

	logging::warning(
			"{} failed ({}:{}) code {} ({})",
			calledFunc,
			relativeFilePath,
			line,
			result,
			errorString ? errorString : "failed to retrieve error string");
}

bool InstrumentationCache::isInstrumented(const CUfunction& function) {
	return !already_instrumented.insert(function).second;
}

ComputeCapability getComputeCapability() {
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	return {deviceProp.major, deviceProp.minor};
}

std::string ComputeCapability::toString() const {
	return fmt::format("{},{}", major, minor);
}

} // namespace util