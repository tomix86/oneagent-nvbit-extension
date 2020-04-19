#include "Configuration.h"

#include "ErrorUtil.h"

#include <cstdlib>
#include <optional>
#include <string_view>
#include <fstream>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

namespace config {

static Configuration configHolder;

#define ENV_VAR_PREFIX "ONEAGENT_NVBIT_EXTENSION_"
constexpr auto confFileEnvVar{ENV_VAR_PREFIX "CONF_FILE" };

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

static void parseConfigFile(Configuration& config, std::ifstream& file) {
    po::options_description options{"Settings"};
    options.add_options()
        ("logfile", po::value<std::filesystem::path>(&config.logFile), "Log file");

    po::variables_map vm;
    po::store(parse_config_file(file, options), vm);
    po::notify(vm);
}

static Configuration load() {
    Configuration config;
    if(const auto val{getEnvVar(confFileEnvVar)}) {
        config.confFile = *val;
    }

    if(!std::filesystem::exists(config.confFile)) {
        throw std::runtime_error{"Configuration file \"" + config.confFile.string() + "\" does not exist"};
    }

    std::ifstream file{config.confFile};
    if(!file) {
        throw std::runtime_error{"Failed to open configuration file \"" + config.confFile.string() + "\" " + util::getLastErrno()};
    }

    parseConfigFile(config, file);
    return config;
}

void initialize() {
    configHolder = load();
}

const Configuration& get() {
    return configHolder;
}

} //namespace config