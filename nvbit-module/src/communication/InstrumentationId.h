#pragma once

#include <stdexcept>
#include <string>

namespace communication {

enum class InstrumentationId { instructions_count, occupancy, memory_access_divergence };

std::string to_string(InstrumentationId id);

bool isInstrumentationIdValid(int id);

} // namespace communication