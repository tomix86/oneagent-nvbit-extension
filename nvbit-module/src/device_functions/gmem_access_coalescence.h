#pragma once

#include <nvbit.h>

#define IMPL_DETAIL_GMEM_ACCESS_COALESCENCE_KERNEL gmemAccessCoalescence

namespace communication {
class MeasurementsPublisher;
}

namespace device::gmem_access_coalescence {

void instrumentKernel(CUcontext context, int is_exit, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher);

}