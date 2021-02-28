#pragma once

#include "InstrumentationId.h"

#include <filesystem>
#include <vector>

namespace communication {

class RuntimeConfiguration {
public:
	void load(const std::filesystem::path& filePath);

	std::vector<InstrumentationId> getInstrumentationFunctions() const;

private:
	std::vector<InstrumentationId> instrumentationFunctions;
};

} // namespace communication