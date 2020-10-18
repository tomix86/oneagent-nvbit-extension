#pragma once

#include <string>
#include <filesystem>

#include "InstrumentationId.h"

namespace communication {

class MeasurementsPublisher {
public:
    void initialize(std::filesystem::path outputDir);

    void publish(InstrumentationId instrumentationFunctionId, const std::string& result);

private:
    std::filesystem::path outputDir;
};

} // namespace communication