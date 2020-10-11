#pragma once

#include "nvbit.h"

namespace communication {
class MeasurementsPublisher;
}

namespace count_instr {

//TODO: de-externalize
extern bool active_region;

void instrumentKernelWithInstructionCounter(CUcontext context, int is_exit, nvbit_api_cuda_t eventId, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}