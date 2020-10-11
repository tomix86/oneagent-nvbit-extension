#include "FunctionToIdMapping.h"

#include "device_functions/functions_registry.h"

#include <array>
#include <algorithm>
#include <stdexcept>

const std::array<std::string, 2> intrumentationFunctionIdToName {
    NAME_OF(INSTRUMENTATION__INSTRUCTIONS_COUNT),
    NAME_OF(INSTRUMENTATION__OCCUPANCY)
};

namespace communication {

std::string nameFromId(int instrumentationFunctionId) {
    return intrumentationFunctionIdToName.at(instrumentationFunctionId);
}

int idFromName(const std::string& instrumentationFunctionName) {
    const auto it{std::find(intrumentationFunctionIdToName.cbegin(), intrumentationFunctionIdToName.cend(), instrumentationFunctionName)};
    if(it == intrumentationFunctionIdToName.end()) {
        throw std::invalid_argument{"Invalid instrumentation function name"};
    }
    
    return std::distance(intrumentationFunctionIdToName.cbegin(), it);
}

bool isIdValid(int instrumentationFunctionId) {
    return instrumentationFunctionId < intrumentationFunctionIdToName.size();
}

} // namespace communication