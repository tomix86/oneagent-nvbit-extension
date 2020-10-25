// Implementation based on "tools/instr_count/instr_count.cu" example from NVBit release package

#include "Configuration.h"
#include "Logger.h"
#include "communication/InstrumentationId.h"
#include "communication/MeasurementsPublisher.h"
#include "count_instrs.h"
#include "util/cuda_utilities.h"
#include "util/preprocessor.h"

#include <cstdint>
#include <mutex>
#include <nvbit.h>

namespace device::count_instr {

__managed__ uint64_t instructionsCount{0};

static void injectInstrumentationRoutine(CUcontext context, CUfunction kernel) {
	static util::InstrumentationCache instrumentationCache;

	// Get related functions of the kernel (device function that can be called by the kernel)
	auto relatedFunctions{nvbit_get_related_functions(context, kernel)};
	relatedFunctions.push_back(kernel);

	for (auto function : relatedFunctions) {
		if (instrumentationCache.isInstrumented(function)) {
			continue;
		}

		for (auto instruction : nvbit_get_instrs(context, function)) {
			if (config::get().verbose) {
				instruction->print();
				instruction->printDecoded();
			}

			nvbit_insert_call(instruction, STRINGIZE(IMPL_DETAIL_COUNT_INSTR_KERNEL), IPOINT_BEFORE);
			if (config::get().exclude_pred_off) {
				nvbit_add_call_arg_guard_pred_val(instruction);
			} else {
				nvbit_add_call_arg_const_val32(instruction, 1);
			}

			nvbit_add_call_arg_const_val32(instruction, config::get().count_warp_level ? 1 : 0);
			nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&instructionsCount));
		}
	}
}

void instrumentKernel(
		CUcontext context,
		int is_exit,
		nvbit_api_cuda_t eventId,
		cuLaunch_params* params,
		communication::MeasurementsPublisher& measurementsPublisher) {
	static std::mutex mutex;

	const auto kernelName{nvbit_get_func_name(context, params->f, config::get().mangled ? 1 : 0)};

	if (!is_exit) {
		logging::info("Instrumenting kernel \"{}\" with {}", kernelName, STRINGIZE(IMPL_DETAIL_COUNT_INSTR_KERNEL));

		mutex.lock(); // prevent multiple kernels from runing concurrently and therefore "corrupting" the instructionsCount variable
		injectInstrumentationRoutine(context, params->f);

		nvbit_enable_instrumented(context, params->f, true);
		instructionsCount = 0;
	} else {
		checkCudaErrors(cudaDeviceSynchronize()); // wait for kernel execution to complete

		int num_ctas{0};
		if (eventId == API_CUDA_cuLaunchKernel_ptsz || eventId == API_CUDA_cuLaunchKernel) {
			const auto kernelLaunchParams = reinterpret_cast<cuLaunchKernel_params*>(params);
			num_ctas = kernelLaunchParams->gridDimX * kernelLaunchParams->gridDimY * kernelLaunchParams->gridDimZ;
		}

		logging::info("kernel \"{}\" - #thread-blocks {}, instructions {}", kernelName, num_ctas, instructionsCount);
		measurementsPublisher.publish(communication::InstrumentationId::instructions_count, std::to_string(instructionsCount));
		mutex.unlock();
	}
}

} // namespace device::count_instr