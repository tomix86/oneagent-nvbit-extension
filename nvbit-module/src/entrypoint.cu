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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "Configuration.h"
#include "Logger.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

void nvbit_at_init() {
    logging::info("NVBit runtime initializing");

    // Make sure all managed variables are allocated on GPU
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    if (!config::get().active_from_start) {
        active_region = false;
    }
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Get the vector of instruction composing the loaded CUFunction "f" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        /* If verbose we print function name and number of" static" instructions
         */
        if (config::get().verbose) {
            logging::debug("inspecting {} - num instrs {}\n", nvbit_get_func_name(ctx, f), instrs.size());
        }

        /* We iterate on the vector of instruction */
        for (auto i : instrs) {
            /* Check if the instruction falls in the interval where we want to
             * instrument */
            if (i->getIdx() >= config::get().instr_begin_interval &&
                i->getIdx() < config::get().instr_end_interval) {
                /* If verbose we print which instruction we are instrumenting
                 * (both offset in the function and SASS string) */
                if (config::get().verbose) {
                    i->print();
                    i->printDecoded();
                }

                /* Insert a call to "count_instrs" before the instruction "i" */
                nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
                if (config::get().exclude_pred_off) {
                    /* pass predicate value */
                    nvbit_add_call_arg_pred_val(i);
                } else {
                    /* pass always true */
                    nvbit_add_call_arg_const_val32(i, 1);
                }

                /* add count warps option */
                nvbit_add_call_arg_const_val32(i, config::get().count_warp_level ? 1 : 0);
                /* add pointer to counter location */
                nvbit_add_call_arg_const_val64(i, (uint64_t)&counter);
            }
        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        cuLaunch_params *p = (cuLaunch_params *)params;

        if (!is_exit) {
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Instrument the function if needed
             * 3. Select if we want to run the instrumented or original
             * version of the kernel
             * 4. Reset the kernel instruction counter */

            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, p->f);

            if (config::get().active_from_start) {
                if (kernel_id >= config::get().start_grid_num && kernel_id < config::get().end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, p->f, true);
            } else {
                nvbit_enable_instrumented(ctx, p->f, false);
            }

            counter = 0;
        } else {
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());
            tot_app_instrs += counter;
            int num_ctas = 0;
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
            }
            logging::info("kernel {} - {} - #thread-blocks {},  kernel instructions {}, total instructions {}", kernel_id++, nvbit_get_func_name(ctx, p->f, config::get().mangled ? 1 : 0), num_ctas, counter, tot_app_instrs);
            pthread_mutex_unlock(&mutex);
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!config::get().active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!config::get().active_from_start) {
            active_region = false;
        }
    }
}

void nvbit_at_term() {
    logging::info("NVBit runtime exiting");
    logging::info("Total app instructions: {}\n", tot_app_instrs);
}
