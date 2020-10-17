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

#include "Configuration.h"
#include "Logger.h"
#include "device_functions/functions_registry.h"
#include "device_functions/count_instrs.h"
#include "device_functions/occupancy.h"
#include "communication/RuntimeConfigurationPoller.h"
#include "communication/MeasurementsPublisher.h"

#include <nvbit_tool.h> // Must be included only once!

#include <boost/algorithm/cxx11/any_of.hpp>

communication::RuntimeConfigurationPoller runtimeConfigPoller;
communication::MeasurementsPublisher measurementsPublisher;

static void instrumentKernelLaunch(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, const std::vector<std::string> & instrumentationFunctions) {
    for(const auto& functionName : instrumentationFunctions) {
        if(functionName == NAME_OF(INSTRUMENTATION__INSTRUCTIONS_COUNT)) {
            count_instr::instrumentKernelWithInstructionCounter(context, is_exit, eventId, params, measurementsPublisher);
        } else if (functionName == NAME_OF(INSTRUMENTATION__OCCUPANCY)) {
            occupancy::instrumentKernelWithOccupancyCounter(context, is_exit, eventId, params, measurementsPublisher);
        } else {
            logging::warning("Unexpected instrumentation function name", functionName);
        }
    }
}

void nvbit_at_init() {
    logging::info("NVBit runtime initializing");

    // Make sure all managed variables are allocated on GPU
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    if (!config::get().active_from_start) {
        count_instr::active_region = false;
    }

    runtimeConfigPoller.initialize(config::get().runtime_config_path, config::get().runtime_config_polling_interval);
    measurementsPublisher.initialize(config::get().measurements_output_dir);
}

void nvbit_at_cuda_event(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, const char* /* name */, void* params, CUresult* /* pStatus */) {
    const auto launchEvents = {API_CUDA_cuLaunch, API_CUDA_cuLaunchKernel_ptsz, API_CUDA_cuLaunchGrid, API_CUDA_cuLaunchGridAsync, API_CUDA_cuLaunchKernel};
    if (boost::algorithm::any_of_equal(launchEvents, eventId)) {
        const auto instrumentationFunctions = runtimeConfigPoller.getConfig().getInstrumentationFunctions();
        instrumentKernelLaunch(context, is_exit, eventId, static_cast<cuLaunch_params*>(params), instrumentationFunctions);
    } else if (eventId == API_CUDA_cuProfilerStart && is_exit) {
        if (!config::get().active_from_start) {
            count_instr::active_region = true;
        }
    } else if (eventId == API_CUDA_cuProfilerStop && is_exit) {
        if (!config::get().active_from_start) {
            count_instr::active_region = false;
        }
    }
}

void nvbit_at_term() {
    logging::info("NVBit runtime exiting");
}