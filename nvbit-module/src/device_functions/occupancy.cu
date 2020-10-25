#include "Configuration.h"
#include "Logger.h"
#include "communication/InstrumentationId.h"
#include "communication/MeasurementsPublisher.h"
#include "occupancy.h"
#include "util/cuda_utilities.h"

#include <cuda_runtime.h>

namespace device::occupancy {

void instrumentKernel(
		CUcontext context,
		int is_exit,
		nvbit_api_cuda_t /*eventId*/,
		cuLaunch_params* params,
		communication::MeasurementsPublisher& measurementsPublisher) {
	if (is_exit) {
		return;
	}

	const auto kernelName = nvbit_get_func_name(context, params->f, config::get().mangled ? 1 : 0);

	logging::info("Instrumenting kernel \"{}\" with occupancy calculation", kernelName);

	int device{};
	checkCudaErrors(cudaGetDevice(&device));

	int maxBlocks{};
	checkCudaErrors(cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMaxBlocksPerMultiprocessor, device));

	// Documentation for cuLaunchKernel_params is available here:
	// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
	const auto kernelLaunchParams = reinterpret_cast<cuLaunchKernel_params*>(params);
	const auto num_ctas = kernelLaunchParams->gridDimX * kernelLaunchParams->gridDimY * kernelLaunchParams->gridDimZ;

	int numBlocks{};
	checkCudaErrors(cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, params->f, num_ctas, kernelLaunchParams->sharedMemBytes));
	const auto occupancy{100 * numBlocks / maxBlocks};

	logging::info("kernel \"{}\" occupancy: {}% ({} / {})", kernelName, occupancy, numBlocks, maxBlocks);
	measurementsPublisher.publish(communication::InstrumentationId::occupancy, std::to_string(occupancy));
}

} // namespace device::occupancy