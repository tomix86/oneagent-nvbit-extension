#pragma once

#include <nvbit.h>

#define IMPL_DETAIL_COUNT_INSTR_KERNEL instructionsCount

namespace communication {
class MeasurementsPublisher;
}

namespace device::count_instr {

void instrumentKernel(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}