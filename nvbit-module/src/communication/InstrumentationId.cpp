#include "InstrumentationId.h"

#include "util/util.h"

#include <boost/algorithm/cxx11/any_of.hpp>

namespace communication {

std::string to_string(InstrumentationId id) {
	switch (id) {
	case InstrumentationId::instructions_count:
		return "instructions_count";
	case InstrumentationId::occupancy:
		return "occupancy";
	case InstrumentationId::gmem_access_coalescence:
		return "gmem_access_coalescence";
	case InstrumentationId::branch_divergence:
		return "branch_divergence";
	default:
		throw std::invalid_argument{"Invalid instrumentation function name"};
	}
}

bool isInstrumentationIdValid(int id) {
	const auto legalIds = {
			util::to_underlying_type(InstrumentationId::instructions_count),
			util::to_underlying_type(InstrumentationId::occupancy),
			util::to_underlying_type(InstrumentationId::gmem_access_coalescence),
			util::to_underlying_type(InstrumentationId::branch_divergence)};
	return boost::algorithm::any_of_equal(legalIds, id);
}

} // namespace communication