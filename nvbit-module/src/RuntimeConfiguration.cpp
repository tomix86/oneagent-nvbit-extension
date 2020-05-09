#include "RuntimeConfiguration.h"

#include "device_functions/functions_registry.h"
#include "ErrorUtil.h"

#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <stdexcept>

using namespace std::string_literals;

const std::unordered_map<int, std::string> intrumentationFunctionIdToName {
    {0, NAME_OF(INSTRUMENTATION__INSTRUCTIONS_COUNT)}
};

namespace config {

void RuntimeConfiguration::load(const std::string& filePath) {
    //TODO: reload only if file contents changed

    std::ifstream input{filePath};
    if(!input) {
        throw std::runtime_error{"opening file failed: " + util::getLastErrno()};
    }

    instrumentationFunctions.clear();

    const auto pid{std::to_string(getpid())};

    std::string buf;
    while(std::getline(input, buf)) {
        std::vector<std::string> tokens;
        boost::split(tokens, buf, boost::is_any_of(":"));
        if(tokens.size() != 2) {
            continue;
        }

        if(tokens.front() != pid) {
            continue;
        }

        try {
            instrumentationFunctions.push_back(std::stoi(tokens.back()));
        } catch(const std::exception&) {
            throw std::runtime_error{"malformed key encountered: " + tokens.back()};
        }
    }
}

std::vector<std::string> RuntimeConfiguration::getInstrumentationFunctions() const {
    std::vector<std::string> result;
    for(auto functionId : instrumentationFunctions) {
        const auto name{intrumentationFunctionIdToName.find(functionId)};
        if(name != intrumentationFunctionIdToName.end()) {
            result.push_back(name->second);
        }
    }

    return result;
}

} // namespace config
