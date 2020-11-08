#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_set>

#define checkCudaErrors(val) util::checkError((val), #val, __FILE__, __LINE__)

namespace util {

void checkError(cudaError_t result, const char* calledFunc, std::string file, int line);
void checkError(CUresult result, const char* calledFunc, std::string file, int line);

class InstrumentationCache {
public:
	bool isInstrumented(const CUfunction& function);

private:
	std::unordered_set<CUfunction> already_instrumented;
};

struct ComputeCapability {
	int major, minor;

	bool operator<(const ComputeCapability& other) const noexcept {
		return major < other.major || minor < other.minor;
	}

	bool operator==(const ComputeCapability& other) const noexcept {
		return !(*this < other || other < *this);
	}

	bool operator!=(const ComputeCapability& other) const noexcept {
		return !(*this == other);
	}

	bool operator>(const ComputeCapability& other) const noexcept {
		return other < *this;
	}

	bool operator<=(const ComputeCapability& other) const noexcept {
		return *this < other || *this == other;
	}

	bool operator>=(const ComputeCapability& other) const noexcept {
		return *this > other || *this == other;
	}

	std::string toString() const;
};

ComputeCapability getComputeCapability();

} // namespace util