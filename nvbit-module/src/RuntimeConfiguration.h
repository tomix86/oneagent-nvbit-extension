#pragma once

#include <vector>
#include <string>

namespace config {

class RuntimeConfiguration {
public:
    void load(const std::string& filePath);

    std::vector<std::string> getInstrumentationFunctions() const;

private:
    std::vector<int> instrumentationFunctions;
};

} // namespace config