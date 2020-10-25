#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <utils/utils.h>

namespace util {

__device__ __forceinline__ bool isFirstActiveThread() {
	const auto activeMask{__ballot_sync(__activemask(), 1)};
	const auto firstActiveThreadLaneid{__ffs(activeMask) - 1};
	const auto threadLaneid{get_laneid()};
	return threadLaneid == firstActiveThreadLaneid;
}

} // namespace util