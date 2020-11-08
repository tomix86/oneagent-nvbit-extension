#include "Configuration.h"
#include "Logger.h"
#include "communication/MeasurementsPublisher.h"
#include "communication/RuntimeConfigurationPoller.h"
#include "device_functions/branch_divergence.h"
#include "device_functions/count_instrs.h"
#include "device_functions/gmem_access_coalescence.h"
#include "device_functions/occupancy.h"

#include <boost/algorithm/cxx11/any_of.hpp>
#include <nvbit_tool.h> // Must be included only once!

communication::RuntimeConfigurationPoller runtimeConfigPoller;
communication::MeasurementsPublisher measurementsPublisher;

static void instrumentKernelLaunch(
		CUcontext context,
		int is_exit,
		nvbit_api_cuda_t eventId,
		cuLaunch_params* params,
		const std::vector<communication::InstrumentationId>& instrumentationFunctions) {
	for (const auto& functionId : instrumentationFunctions) {
		switch (functionId) {
		case communication::InstrumentationId::instructions_count:
			device::count_instr::instrumentKernel(context, is_exit, eventId, params, measurementsPublisher);
			break;
		case communication::InstrumentationId::occupancy:
			device::occupancy::instrumentKernel(context, is_exit, eventId, params, measurementsPublisher);
			break;
		case communication::InstrumentationId::gmem_access_coalescence:
			device::gmem_access_coalescence::instrumentKernel(context, is_exit, params, measurementsPublisher);
			break;
		case communication::InstrumentationId::branch_divergence:
			device::branch_divergence::instrumentKernel(context, is_exit, params, measurementsPublisher);
			break;
		default:
			break;
		}
	}
}

void nvbit_at_init() {
	logging::info("NVBit runtime initializing, version: {}", NVBIT_VERSION);

	// Make sure all managed variables are allocated on GPU
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

	runtimeConfigPoller.initialize(config::get().runtime_config_path, config::get().runtime_config_polling_interval);
	measurementsPublisher.initialize(config::get().measurements_output_dir);
}

void nvbit_at_cuda_event(
		CUcontext context, int is_exit, nvbit_api_cuda_t eventId, const char* /* name */, void* params, CUresult* /* pStatus */) {
	const auto launchEvents = {
			API_CUDA_cuLaunch, API_CUDA_cuLaunchKernel_ptsz, API_CUDA_cuLaunchGrid, API_CUDA_cuLaunchGridAsync, API_CUDA_cuLaunchKernel};
	if (boost::algorithm::any_of_equal(launchEvents, eventId)) {
		const auto instrumentationFunctions = runtimeConfigPoller.getConfig().getInstrumentationFunctions();
		instrumentKernelLaunch(context, is_exit, eventId, static_cast<cuLaunch_params*>(params), instrumentationFunctions);
	}
}

void nvbit_at_term() {
	logging::info("NVBit runtime exiting");
}