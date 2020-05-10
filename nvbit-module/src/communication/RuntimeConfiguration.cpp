#include "RuntimeConfiguration.h"

#include "../ErrorUtil.h"
#include "FunctionToIdMapping.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <stdexcept>

using namespace std::string_literals;

namespace communication {

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
        if(isIdValid(functionId)) {
            result.push_back(nameFromId(functionId));
        }
    }

    return result;
}

} // namespace communication