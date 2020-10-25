#pragma once

#include <nvbit.h>

#define IMPL_DETAIL_BRANCH_DIVERGENCE_KERNEL branchDivergence

namespace communication {
class MeasurementsPublisher;
}

namespace device::branch_divergence {

void instrumentKernel(CUcontext context, int is_exit, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}