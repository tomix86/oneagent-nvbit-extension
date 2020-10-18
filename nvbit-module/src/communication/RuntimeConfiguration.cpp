#include "RuntimeConfiguration.h"

#include "util/ErrorUtil.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <stdexcept>

using namespace std::string_literals;

namespace communication {

//TODO: thread safety
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
            std::vector<std::string> functionIds;
            boost::split(functionIds, tokens.back(), boost::is_any_of(","));
            for(const auto& id : functionIds) {
                instrumentationFunctions.push_back(std::stoi(id));
            }
        } catch(const std::exception&) {
            throw std::runtime_error{"malformed key encountered: " + tokens.back()};
        }
    }
}

std::vector<InstrumentationId> RuntimeConfiguration::getInstrumentationFunctions() const {
    std::vector<InstrumentationId> result;
    for(auto functionId : instrumentationFunctions) {
        if(isInstrumentationIdValid(functionId)) {
            result.push_back(InstrumentationId{functionId});
        }
    }

    return result;
}

} // namespace communication
