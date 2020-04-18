#include "Configuration.h"

#include <cstdlib>
#include <optional>
#include <string_view>
#include <sstream>

namespace config {

#define ENV_VAR_PREFIX "ONEAGENT_NVBIT_EXTENSION_"
constexpr auto confFileEnvVar{ENV_VAR_PREFIX "CONF_FILE" };
constexpr auto logFileEnvVar{ENV_VAR_PREFIX "LOG_FILE" };

void Configuration::print(std::function<void(std::string_view line)> linePrinter) const {
    linePrinter("Config file: " + confFile.string());
    linePrinter("Log file: " + logFile.string());
}

static std::optional<std::string> getEnvVar(std::string_view name) {
    if (const auto value{std::getenv(name.data())}) {
        return value;
    } 

    return std::nullopt;
}

static Configuration load() {
    const auto confFile{getEnvVar(confFileEnvVar).value_or("nvbit-module.conf")};
    if(std::filesystem::exists(confFile)) {
        //TODO: Process config here
    }

    return {
        confFile,
        getEnvVar(logFileEnvVar).value_or("nvbit-module.log")
    };
}

const Configuration& get() {
    static std::optional<Configuration> config;
    if(!config) {
        config = load();
    }

    return *config;
}

} //namespace config