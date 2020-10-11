/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdint>
#include <cuda_runtime.h>

#include "utils/utils.h"

#include "functions_registry.h"

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