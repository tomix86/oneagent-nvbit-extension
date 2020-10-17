#include "functions_registry.h"

#include <cstdint>
#include <utils/utils.h>

#include <cuda_runtime.h>

namespace count_instr {

extern "C" __device__ __noinline__ void INSTRUMENTATION__INSTRUCTIONS_COUNT(int predicate, int count_warp_level, uint64_t pcounter) {
    const int active_mask = ballot(1); /* all the active threads will compute the active mask (ballot() is implemented in utils/utils.h)*/  
    const int laneid = get_laneid(); /* each thread will get a lane id (get_lane_id is implemented in utils/utils.h) */
    const int first_laneid = __ffs(active_mask) - 1; /* get the id of the first active thread */
    if (first_laneid != laneid) {  /* only the first active thread will perform the atomic */
        return;
    }

    const int predicate_mask = ballot(predicate); /* compute the predicate mask */
    const int num_threads = __popc(predicate_mask); /* count all the active thread */
    if (count_warp_level) {
        if (num_threads > 0) { /* num threads can be zero when accounting for predicates off */
            atomicAdd(reinterpret_cast<unsigned long long*>(pcounter), 1);
        }
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(pcounter), num_threads);
    }
}

}