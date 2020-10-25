// Implementation based on "tools/mem_printf/mem_printf.cu" example from NVBit release package and Listing 8 from the paper: "Oreste Villa,
// Mark Stephenson, David Nellans, and Stephen W. Keckler. 2019. NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs. In
// Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO '52). Association for Computing Machinery,
// New York, NY, USA, 372–383. DOI:https://doi.org/10.1145/3352460.3358307"

#include "Configuration.h"
#include "Logger.h"
#include "communication/InstrumentationId.h"
#include "communication/MeasurementsPublisher.h"
#include "memory_access_divergence.h"
#include "util/cuda_utilities.h"
#include "util/preprocessor.h"

#include <boost/range/adaptor/filtered.hpp>
#include <vector>

using boost::adaptors::filtered;

namespace device::memory_access_divergence {

__managed__ float uniqueCacheLinesAccesses{0};
__managed__ int memoryAccessesCount{0};

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
			nvbit_insert_call(instruction, STRINGIZE(IMPL_DETAIL_MEM_ACCESS_DIVERGENCE_KERNEL), IPOINT_BEFORE);
			nvbit_add_call_arg_guard_pred_val(instruction);
			nvbit_add_call_arg_mref_addr64(instruction, mref_idx++);
			nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&uniqueCacheLinesAccesses));
			nvbit_add_call_arg_const_val64(instruction, reinterpret_cast<uint64_t>(&memoryAccessesCount));
		}
	}
}

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
	logging::info("Instrumenting kernel \"{}\" with {}", kernelName, STRINGIZE(IMPL_DETAIL_MEM_ACCESS_DIVERGENCE_KERNEL));

	injectInstrumentationRoutine(context, kernel);

	const auto result{uniqueCacheLinesAccesses / memoryAccessesCount};
	logging::info("kernel \"{}\", average cache lines requests per memory instruction {}", kernelName, result);
	measurementsPublisher.publish(communication::InstrumentationId::memory_access_divergence, std::to_string(result));
}

} // namespace device::memory_access_divergence