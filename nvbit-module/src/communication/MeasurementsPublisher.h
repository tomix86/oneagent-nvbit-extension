#pragma once

#include <string>

#include "FunctionToIdMapping.h"

namespace communication {

class MeasurementsPublisher {
public:
    void initialize(std::string outputDir);

    void publish(InstrumentationId instrumentationFunctionId, const std::string& result);

private:
    std::string outputDir;
};

} // namespace communication