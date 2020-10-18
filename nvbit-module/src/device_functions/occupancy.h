#pragma once

#include <nvbit.h>

namespace communication {
class MeasurementsPublisher;
}

namespace device::occupancy {

void instrumentKernel(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}