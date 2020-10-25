#pragma once

#include <nvbit.h>

#define IMPL_DETAIL_MEM_ACCESS_DIVERGENCE_KERNEL memoryAccessDivergence

namespace communication {
class MeasurementsPublisher;
}

namespace device::memory_access_divergence {

void instrumentKernel(CUcontext context, int is_exit, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}