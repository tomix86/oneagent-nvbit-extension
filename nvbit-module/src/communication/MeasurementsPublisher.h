#pragma once

#include "InstrumentationId.h"

#include <filesystem>
#include <string>

namespace communication {

class MeasurementsPublisher {
public:
	void initialize(std::filesystem::path outputDir);

	void publish(InstrumentationId instrumentationFunctionId, const std::string& result);

private:
	std::filesystem::path outputDir;
};

} // namespace communication