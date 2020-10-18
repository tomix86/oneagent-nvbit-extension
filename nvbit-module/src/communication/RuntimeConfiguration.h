#pragma once

#include <vector>
#include <filesystem>

#include "InstrumentationId.h"

namespace communication {

class RuntimeConfiguration {
public:
    void load(const std::filesystem::path& filePath);

    std::vector<InstrumentationId> getInstrumentationFunctions() const;

private:
    std::vector<int> instrumentationFunctions;
};

} // namespace communication