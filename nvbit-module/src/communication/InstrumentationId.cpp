#include "InstrumentationId.h"
#include "util/util.h"

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/count_if.hpp>
#include <stdexcept>

namespace communication {

using Id = InstrumentationId;

std::string to_string(Id id) {
	switch (id) {
	case Id::instructions_count:
		return "instructions_count";
	case Id::occupancy:
		return "occupancy";
	case Id::gmem_access_coalescence:
		return "gmem_access_coalescence";
	case Id::branch_divergence:
		return "branch_divergence";
	default:
		throw std::invalid_argument{"Invalid instrumentation function name"};
	}
}

bool isInstrumentationIdValid(int id) {
	const auto legalIds = {
			util::to_underlying_type(Id::instructions_count),
			util::to_underlying_type(Id::occupancy),
			util::to_underlying_type(Id::gmem_access_coalescence),
			util::to_underlying_type(Id::branch_divergence)};
	return boost::algorithm::any_of_equal(legalIds, id);
}

bool isInstrumentationSetValid(const std::vector<Id>& set) {
	return boost::range::count_if(set, [](auto id) {
		const auto injectionRoutines = {Id::instructions_count, Id::gmem_access_coalescence, Id::branch_divergence};
		return boost::algorithm::any_of_equal(injectionRoutines, id);
	}) <= 1;
}

} // namespace communication