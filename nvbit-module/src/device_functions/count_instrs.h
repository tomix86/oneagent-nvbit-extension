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



#include <cuda_runtime.h>
//TODO: move to utility header
#define checkCudaErrors(val) checkError((val), #val, __FILE__, __LINE__)
void checkError(cudaError_t result, const char* calledFunc, std::string file, int line);