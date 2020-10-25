#include "Configuration.h"
#include "Logger.h"
#include "branch_divergence.h"
#include "communication/InstrumentationId.h"
#include "communication/MeasurementsPublisher.h"
#include "util/cuda_utilities.h"
#include "util/preprocessor.h"

namespace device::branch_divergence {

void instrumentKernel(
		CUcontext context, int is_exit, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher) {
	if (is_exit) {
		return;
	}

	const auto kernel{params->f};

	static util::InstrumentationCache instrumentationCache;
	if (instrumentationCache.isInstrumented(kernel)) {
		return;
	}

	const auto kernelName{nvbit_get_func_name(context, kernel, config::get().mangled ? 1 : 0)};
	logging::info("Instrumenting kernel \"{}\" with {}", kernelName, STRINGIZE(IMPL_DETAIL_BRANCH_DIVERGENCE_KERNEL));

	// TODO: Implement

	const auto result{100};
	logging::info("kernel \"{}\", divergent branches: {}%", kernelName, result);
	measurementsPublisher.publish(communication::InstrumentationId::branch_divergence, std::to_string(result));
}

} // namespace device::branch_divergence