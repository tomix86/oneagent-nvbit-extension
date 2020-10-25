// Implementation based on Listing 8 from the paper: "Oreste Villa, Mark Stephenson, David Nellans, and Stephen W. Keckler. 2019. NVBit: A
// Dynamic Binary Instrumentation Framework for NVIDIA GPUs. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on
// Microarchitecture< (MICRO '52). Association for Computing Machinery, New York, NY, USA, 372–383.
// DOI:https://doi.org/10.1145/3352460.3358307"

#include "device_utility_functions.h"
#include "memory_access_divergence.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace device::memory_access_divergence {

// __match_any_sync() is supported for CC 7.0 and higher
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-match-functions
#if __CUDA_ARCH__ >= 700

extern "C" __device__ __noinline__ void IMPL_DETAIL_MEM_ACCESS_DIVERGENCE_KERNEL(
		int predicate, uint64_t address, uint64_t uniqueCacheLinesAccesses, uint64_t memoryAccessesCount) {
	if (!predicate) {
		return;
	}

	if (util::isFirstActiveThread()) {
		atomicAdd(reinterpret_cast<int*>(memoryAccessesCount), 1);
	}

	constexpr auto cacheLineSize{7}; // log2(128)
	const auto cacheLineAddress{address >> cacheLineSize};

	const auto activeMask{__ballot_sync(__activemask(), 1)};
	const auto threadsAccessingCacheLine{__popc(__match_any_sync(activeMask, cacheLineAddress))};
	// each thread contributes proportionally to the cache line counter, see
	// https://github.com/NVlabs/NVBit/issues/24#issuecomment-661176067
	atomicAdd(reinterpret_cast<float*>(uniqueCacheLinesAccesses), 1.f / threadsAccessingCacheLine);
}

#endif // __CUDA_ARCH__ >= 700

} // namespace device::memory_access_divergence