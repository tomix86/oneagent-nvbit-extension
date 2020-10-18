#include "count_instrs.h"

#include "Configuration.h"
#include "Logger.h"
#include "communication/FunctionToIdMapping.h"
#include "communication/MeasurementsPublisher.h"
#include "util/cuda_utilities.h"
#include "util/preprocessor.h"

#include <nvbit.h>
#include <utils/utils.h>

#include <cstdint>
#include <unordered_set>
#include <mutex>
#include <cuda_runtime.h>

namespace device::count_instr {

__managed__ uint64_t counter = 0; // kernel instruction counter, updated by the GPU
uint64_t tot_app_instrs = 0; // total instruction counter, maintained in system memory, incremented by "counter" every time a kernel completes

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
            // If verbose we print which instruction we are instrumenting (both offset in the function and SASS string)
            if (config::get().verbose) {
                instruction->print();
                instruction->printDecoded();
            }

            nvbit_insert_call(instruction, instrumentationFunction.c_str(), IPOINT_BEFORE); // Insert a call to instrumentation routine before the instruction
            if (config::get().exclude_pred_off) {
                nvbit_add_call_arg_guard_pred_val(instruction); // pass predicate value
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
void instrumentKernel(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher) {
    static uint32_t kernel_id = 0; // kernel id counter, maintained in system memory
    static std::mutex mutex; // used to prevent multiple kernels to run concurrently and therefore to "corrupt" the counter variable

    const auto kernelName = nvbit_get_func_name(context, params->f, config::get().mangled ? 1 : 0);

    if (!is_exit) {
        logging::info("Instrumenting kernel {} with {} function", kernelName, STRINGIZE(IMPL_DETAIL_COUNT_INSTR_KERNEL));

        mutex.lock();
        instrumentFunctionIfNeeded(context, params->f, STRINGIZE(IMPL_DETAIL_COUNT_INSTR_KERNEL));

        nvbit_enable_instrumented(context, params->f, true);
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
        measurementsPublisher.publish(communication::InstrumentationId::instructions_count, std::to_string(counter));
        mutex.unlock();
    }
}

}