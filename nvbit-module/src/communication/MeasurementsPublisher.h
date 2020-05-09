#pragma once

#include <string>

namespace communication {

class MeasurementsPublisher {
public:
    void initialize(std::string outputDir);

    void publish(const std::string& instrumentationFunctionName, const std::string& result);

private:
    std::string outputDir;
};

} // namespace communication