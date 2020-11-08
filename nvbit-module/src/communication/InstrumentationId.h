#pragma once

#include <stdexcept>
#include <string>

namespace communication {

enum class InstrumentationId { instructions_count, occupancy, gmem_access_coalescence, branch_divergence };

std::string to_string(InstrumentationId id);

bool isInstrumentationIdValid(int id);

} // namespace communication