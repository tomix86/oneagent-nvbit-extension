// Implementation based on Listing 8 from the paper: "Oreste Villa, Mark Stephenson, David Nellans, and Stephen W. Keckler. 2019. NVBit: A
// Dynamic Binary Instrumentation Framework for NVIDIA GPUs. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on
// Microarchitecture (MICRO '52). Association for Computing Machinery, New York, NY, USA, 372–383.
// DOI:https://doi.org/10.1145/3352460.3358307"

#include "device_utility_functions.h"
#include "gmem_access_coalescence.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace device::gmem_access_coalescence {

extern "C" __device__ __noinline__ void IMPL_DETAIL_GMEM_ACCESS_COALESCENCE_KERNEL(
		int predicate, uint64_t address, uint64_t uniqueCacheLinesAccesses, uint64_t memoryAccessesCount) {
	if (!predicate) {
		return;
	}

	if (util::isFirstActiveThread()) {
		atomicAdd(reinterpret_cast<unsigned long long*>(memoryAccessesCount), 1);
	}

	constexpr auto cacheLineSize{7}; // log2(128)
	const auto cacheLineAddress{address >> cacheLineSize};
	const auto activeMask{__ballot_sync(__activemask(), 1)};
	const auto threadsAccessingCacheLine{__popc(util::matchAnySync(activeMask, cacheLineAddress))};

	// each thread contributes proportionally to the cache line counter, see
	// https://github.com/NVlabs/NVBit/issues/24#issuecomment-661176067
	atomicAdd(reinterpret_cast<float*>(uniqueCacheLinesAccesses), 1.f / threadsAccessingCacheLine);
}

} // namespace device::gmem_access_coalescence