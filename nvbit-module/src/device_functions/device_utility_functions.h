#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <utils/utils.h>

namespace util {

__device__ __forceinline__ bool isFirstActiveThread() {
	const auto activeMask{__ballot_sync(__activemask(), 1)};
	const auto firstActiveThreadLaneid{__ffs(activeMask) - 1};
	const auto threadLaneid{get_laneid()};
	return threadLaneid == firstActiveThreadLaneid;
}

__device__ __forceinline__ void raiseError(const char* source, const char* text) {
	fprintf(stderr, "[Device code assertion] %s: %s \n", source, text);
	assert(false);
	__trap();
}

// Based on https://stackoverflow.com/questions/59879285/whats-the-alternative-for-match-any-sync-on-compute-capability-6
__device__ __forceinline__ int matchAnySync(unsigned mask, unsigned long long value) {
// __match_any_sync() is supported for CC 7.0 and higher
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-match-functions
#if __CUDA_ARCH__ >= 700
	return __match_any_sync(mask, value);
#else
	for (int i{0}; i < warpSize; i++) {
		if ((1U << i) & mask) {
			const auto threadValue{__shfl_sync(mask, value, i)};
			const auto newMask{__ballot_sync(mask, threadValue == value)};
			if (i == (threadIdx.x & (warpSize - 1))) {
				mask = newMask;
			}
		}
	}

	return mask;
#endif // __CUDA_ARCH__ >= 700
}

} // namespace util