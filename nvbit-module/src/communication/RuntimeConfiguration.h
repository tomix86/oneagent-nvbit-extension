#pragma once

#include <vector>
#include <string>

namespace communication {

class RuntimeConfiguration {
public:
    void load(const std::string& filePath);

    std::vector<std::string> getInstrumentationFunctions() const;

private:
    std::vector<int> instrumentationFunctions;
};

} // namespace communication