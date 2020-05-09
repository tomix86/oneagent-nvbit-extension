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

#include "nvbit_tool.h"
#include "nvbit.h"
#include "Configuration.h"
#include "Logger.h"
#include "device_functions/functions_registry.h"
#include "communication/RuntimeConfigurationPoller.h"
#include "communication/MeasurementsPublisher.h"

#include <pthread.h>
#include <cstdint>
#include <unordered_set>
#include <boost/algorithm/cxx11/any_of.hpp>

__managed__ uint64_t counter = 0; // kernel instruction counter, updated by the GPU
uint64_t tot_app_instrs = 0; // total instruction counter, maintained in system memory, incremented by "counter" every time a kernel completes
bool active_region = true; // used to select region of insterest when active from start is off

communication::RuntimeConfigurationPoller runtimeConfigPoller;
communication::MeasurementsPublisher measurementsPublisher;

#define checkCudaErrors(val) checkError((val), #val, __FILE__, __LINE__)
void checkError(cudaError_t result, const char* calledFunc, std::string file, int line) {
    if (!result) { return; }
    const auto relativeFilePath = file.substr(file.rfind("src/"));
    logging::info("{} failed ({}:{}) code {} ({})", calledFunc, relativeFilePath, line, result, cudaGetErrorString(result));
}

// Test whether val ∈ [low; high)
template <typename T>
bool is_in_range(const T& val, const T& low, const T& high) {
    return val >= low && val < high;
}

static void instrumentFunctionIfNeeded(CUcontext context, CUfunction func, const std::string& instrumentationFunction) {
    static std::unordered_set<CUfunction> already_instrumented;

    auto relatedFunctions = nvbit_get_related_functions(context, func); // Get related functions of the kernel (device function that can be called by the kernel)
    relatedFunctions.push_back(func); // add kernel itself to the related function vector

    for (auto function : relatedFunctions) {
        if (!already_instrumented.insert(function).second) {
            continue;
        }

        const auto instructions = nvbit_get_instrs(context, function);

        if (config::get().verbose) {
            logging::debug("inspecting {} - instructions count: {}\n", nvbit_get_func_name(context, function), instructions.size());
        }

        for (auto instruction : instructions) {
            if(!is_in_range(instruction->getIdx(), config::get().instr_begin_interval, config::get().instr_end_interval)) {
                continue;
            }

            // If verbose we print which instruction we are instrumenting (both offset in the function and SASS string)
            if (config::get().verbose) {
                instruction->print();
                instruction->printDecoded();
            }

            nvbit_insert_call(instruction, instrumentationFunction.c_str(), IPOINT_BEFORE); // Insert a call to instrumentation routine before the instruction
            if (config::get().exclude_pred_off) {
                nvbit_add_call_arg_pred_val(instruction); // pass predicate value
            } else {
                nvbit_add_call_arg_const_val32(instruction, 1); // pass always true
            }

            nvbit_add_call_arg_const_val32(instruction, config::get().count_warp_level ? 1 : 0);  // add count warps option
            nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&counter)); // add pointer to counter location
        }
    }
}

/* 
if we are entering in a kernel launch:
    1. Lock the mutex to prevent multiple kernels to run concurrently(overriding the counter) in case the user application does that
    2. Instrument the function if needed
    3. Select if we want to run the instrumented or original version of the kernel
    4. Reset the kernel instruction counter
if we are exiting a kernel launch:
    1. Wait until the kernel is completed using cudaDeviceSynchronize()
    2. Get number of thread blocks in the kernel
    3. Print the thread instruction counters
    4. Release the lock
*/
static void instrumentKernelLaunch(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, const std::string& instrumentationFunction) {
    static uint32_t kernel_id = 0; // kernel id counter, maintained in system memory
    static pthread_mutex_t mutex; // used to prevent multiple kernels to run concurrently and therefore to "corrupt" the counter variable

    const auto kernelName = nvbit_get_func_name(context, params->f, config::get().mangled ? 1 : 0);

    if (!is_exit) {
        logging::info("Instrumenting kernel {} with {} function", kernelName, instrumentationFunction);

        pthread_mutex_lock(&mutex);
        instrumentFunctionIfNeeded(context, params->f, instrumentationFunction);

        if (config::get().active_from_start) {
            active_region = is_in_range(kernel_id, config::get().start_grid_num, config::get().end_grid_num);
        }

        nvbit_enable_instrumented(context, params->f, active_region);
        counter = 0;
    } else {
        checkCudaErrors(cudaDeviceSynchronize());
        tot_app_instrs += counter;
        
        int num_ctas = 0;
        if (eventId == API_CUDA_cuLaunchKernel_ptsz || eventId == API_CUDA_cuLaunchKernel) {
            const auto kernelLaunchParams = reinterpret_cast<cuLaunchKernel_params*>(params);
            num_ctas = kernelLaunchParams->gridDimX * kernelLaunchParams->gridDimY * kernelLaunchParams->gridDimZ;
        }

        logging::info("kernel {} - {} - #thread-blocks {},  kernel instructions {}, total instructions {}", kernel_id++, kernelName, num_ctas, counter, tot_app_instrs);
        measurementsPublisher.publish(instrumentationFunction, std::to_string(counter));
        pthread_mutex_unlock(&mutex);
    }
}

void nvbit_at_init() {
    logging::info("NVBit runtime initializing");

    // Make sure all managed variables are allocated on GPU
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    if (!config::get().active_from_start) {
        active_region = false;
    }

    runtimeConfigPoller.initialize(config::get().runtime_config_path, std::chrono::seconds{config::get().runtime_config_polling_interval});
    measurementsPublisher.initialize(config::get().measurements_output_dir);
}

void nvbit_at_cuda_event(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, const char* /* name */, void* params, CUresult* /* pStatus */) {
    const auto launchEvents = {API_CUDA_cuLaunch, API_CUDA_cuLaunchKernel_ptsz, API_CUDA_cuLaunchGrid, API_CUDA_cuLaunchGridAsync, API_CUDA_cuLaunchKernel};
    if (boost::algorithm::any_of_equal(launchEvents, eventId)) {
        const auto instrumentationFunctions = runtimeConfigPoller.getConfig().getInstrumentationFunctions();
        if(!instrumentationFunctions.empty()) {
            instrumentKernelLaunch(context, is_exit, eventId, static_cast<cuLaunch_params*>(params), instrumentationFunctions.front());
        }
    } else if (eventId == API_CUDA_cuProfilerStart && is_exit) {
        if (!config::get().active_from_start) {
            active_region = true;
        }
    } else if (eventId == API_CUDA_cuProfilerStop && is_exit) {
        if (!config::get().active_from_start) {
            active_region = false;
        }
    }
}

void nvbit_at_term() {
    logging::info("NVBit runtime exiting");
    logging::info("Total app instructions: {}", tot_app_instrs);
}