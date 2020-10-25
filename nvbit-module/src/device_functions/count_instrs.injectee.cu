// Implementation based on "tools/instr_count/instr_count.cu" example from NVBit release package

#include "count_instrs.h"
#include "device_utility_functions.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace device::count_instr {

extern "C" __device__ __noinline__ void IMPL_DETAIL_COUNT_INSTR_KERNEL(int predicate, int countWarpLevel, uint64_t counter) {
	if (!util::isFirstActiveThread()) {
		return;
	}

	const auto predicateMask{__ballot_sync(__activemask(), predicate)};
	const auto activeThreads{__popc(predicateMask)};
	if (countWarpLevel) {
		if (activeThreads > 0) {
			atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1);
		}
	} else {
		atomicAdd(reinterpret_cast<unsigned long long*>(counter), activeThreads);
	}
}

} // namespace device::count_instr