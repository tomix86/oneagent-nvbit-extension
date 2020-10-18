#pragma once

#include <vector>
#include <string>

#include "InstrumentationId.h"

namespace communication {

class RuntimeConfiguration {
public:
    void load(const std::string& filePath);

    std::vector<InstrumentationId> getInstrumentationFunctions() const;

private:
    std::vector<int> instrumentationFunctions;
};

} // namespace communication