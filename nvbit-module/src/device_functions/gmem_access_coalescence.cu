// Implementation based on "tools/mem_printf/mem_printf.cu" example from NVBit release package and Listing 8 from the paper: "Oreste Villa,
// Mark Stephenson, David Nellans, and Stephen W. Keckler. 2019. NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs. In
// Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO '52). Association for Computing Machinery,
// New York, NY, USA, 372–383. DOI:https://doi.org/10.1145/3352460.3358307"

#include "Configuration.h"
#include "Logger.h"
#include "communication/InstrumentationId.h"
#include "communication/MeasurementsPublisher.h"
#include "gmem_access_coalescence.h"
#include "util/cuda_utilities.h"
#include "util/preprocessor.h"

#include <boost/range/adaptor/filtered.hpp>
#include <mutex>
#include <vector>

using boost::adaptors::filtered;

namespace device::gmem_access_coalescence {

__managed__ float uniqueCacheLinesAccesses{1};
__managed__ int memoryAccessesCount{1};

static auto getInstructionOperands(Instr* instruction) {
	std::vector<const InstrType::operand_t*> operands;
	for (int n{0}; n < instruction->getNumOperands(); n++) {
		operands.push_back(instruction->getOperand(n));
	}

	return operands;
}

static bool isGlobal(Instr* instruction) {
	return instruction->getMemorySpace() == InstrType::MemorySpace::GLOBAL;
}

static bool isMRef(const InstrType::operand_t* operand) {
	return operand->type == InstrType::OperandType::MREF;
}

static void injectInstrumentationRoutine(CUcontext context, CUfunction kernel) {
	for (auto instruction : nvbit_get_instrs(context, kernel) | filtered(isGlobal)) {
		if (config::get().verbose) {
			instruction->print();
			instruction->printDecoded();
		}

		int mref_idx{0};
		for (auto operand : getInstructionOperands(instruction) | filtered(isMRef)) {
			nvbit_insert_call(instruction, STRINGIZE(IMPL_DETAIL_GMEM_ACCESS_COALESCENCE_KERNEL), IPOINT_BEFORE);
			nvbit_add_call_arg_guard_pred_val(instruction);
			nvbit_add_call_arg_mref_addr64(instruction, mref_idx++);
			nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&uniqueCacheLinesAccesses));
			nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&memoryAccessesCount));
		}
	}
}

void instrumentKernel(
		CUcontext context, int is_exit, cuLaunch_params* params, communication::MeasurementsPublisher& measurementsPublisher) {
	static std::mutex mutex;

	const auto kernel{params->f};
	const auto kernelName{nvbit_get_func_name(context, kernel, config::get().mangled ? 1 : 0)};

	constexpr util::ComputeCapability minimumRequiredCC{7, 0};
	if (const auto cc{util::getComputeCapability()}; cc < minimumRequiredCC) {
		logging::debug(
				"Skipping instrumentation of kernel \"{}\" with {} as minimum compute capability requirements are not met ({} vs {})",
				kernelName,
				STRINGIZE(IMPL_DETAIL_GMEM_ACCESS_COALESCENCE_KERNEL),
				cc.toString(),
				minimumRequiredCC.toString());
		return;
	}

	if (!is_exit) {
		static util::InstrumentationCache instrumentationCache;
		if (instrumentationCache.isInstrumented(kernel)) {
			return;
		}

		logging::info("Instrumenting kernel \"{}\" with {}", kernelName, STRINGIZE(IMPL_DETAIL_GMEM_ACCESS_COALESCENCE_KERNEL));

		mutex.lock(); // prevent multiple kernels from runing concurrently and therefore "corrupting" the data variables
		injectInstrumentationRoutine(context, kernel);
		nvbit_enable_instrumented(context, kernel, true);
		uniqueCacheLinesAccesses = 1;
		memoryAccessesCount = 1;
	} else {
		checkCudaErrors(cudaDeviceSynchronize()); // wait for kernel execution to complete

		const auto cacheLinesPerMemoryInstruction{uniqueCacheLinesAccesses / memoryAccessesCount};
		const auto coalescenceFactor{100 / cacheLinesPerMemoryInstruction};
		logging::info(
				"kernel \"{}\", global memory access coalescence factor {}%  (cache lines accessed: {}; memory instructions: {})",
				kernelName,
				coalescenceFactor,
				uniqueCacheLinesAccesses,
				memoryAccessesCount);
		measurementsPublisher.publish(communication::InstrumentationId::gmem_access_coalescence, std::to_string(coalescenceFactor));

		mutex.unlock();
	}
}

} // namespace device::gmem_access_coalescence