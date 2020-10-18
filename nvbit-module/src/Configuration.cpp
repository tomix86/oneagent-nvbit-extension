#include "Configuration.h"

#include "util/ErrorUtil.h"

#include <cstdlib>
#include <optional>
#include <fstream>
#include <filesystem>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

template <typename Rep, typename Period>
std::istream& operator>>(std::istream& str, std::chrono::duration<Rep, Period>& value) {
    Rep tmp;
    str >> tmp;
    value = std::chrono::duration<Rep, Period>{tmp};
    return str;
}

namespace config {

static Configuration configHolder;

#define ENV_VAR_PREFIX "ONEAGENT_NVBIT_EXTENSION_"
constexpr auto confFileEnvVar{ENV_VAR_PREFIX "CONF_FILE" };

static std::string to_string(bool value) {
    return (value ? "true" : "false");
}

void Configuration::print(std::function<void(std::string line)> linePrinter) const {
    linePrinter("Config file: " + confFile);
    linePrinter("Log file: " + logFile);
    linePrinter("verbose: " + to_string(verbose));
    linePrinter("count_warp_level: " + to_string(count_warp_level));
    linePrinter("exclude_pred_off: " + to_string(exclude_pred_off));
    linePrinter("mangled: " + to_string(mangled));
    linePrinter("runtime config path: " + runtime_config_path);
    linePrinter("runtime config polling interval: " + std::to_string(runtime_config_polling_interval.count()) + "s");
    linePrinter("measurements output directory: " + measurements_output_dir);
}

static std::optional<std::string> getEnvVar(std::string name) {
    if (const auto value{std::getenv(name.data())}) {
        return value;
    }

    return std::nullopt;
}

static void parseConfigFile(Configuration& config, std::ifstream& file) {
    po::options_description options{"Settings"};
    options.add_options()
        ("logfile", po::value<std::string>(&config.logFile), "Log file")
        ("verbose", po::value<bool>(&config.verbose), "Enable verbosity inside the tool")
        ("count_warp_level", po::value<bool>(&config.count_warp_level), "Count warp level or thread level instructions")
        ("exclude_pred_off", po::value<bool>(&config.exclude_pred_off), "Exclude predicated off instruction from count")
        ("mangled_names", po::value<bool>(&config.mangled), "Print kernel names mangled or not")
        ("runtime_config_path", po::value<std::string>(&config.runtime_config_path), "Path to runtime configuration file")
        ("runtime_config_polling_interval", po::value<std::chrono::seconds>(&config.runtime_config_polling_interval), "Runtime configuration file polling interval")
        ("measurements_output_dir", po::value<std::string>(&config.measurements_output_dir), "Path to results output directory")
    ;

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
        throw std::runtime_error{"Configuration file \"" + config.confFile + "\" does not exist"};
    }

    std::ifstream file{config.confFile};
    if(!file) {
        throw std::runtime_error{"Failed to open configuration file \"" + config.confFile + "\" " + util::getLastErrno()};
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