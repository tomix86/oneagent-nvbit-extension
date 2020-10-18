#pragma once

#include <string> 
#include <cstdint>
#include <chrono>
#include <functional>

//TODO: fields storing paths should be of std::filesystem::path type, unfortunately nvcc seems to be missing support for <filesystem>

namespace config {
struct Configuration {
    std::string confFile{"nvbit-module.conf"};
    std::string logFile{"nvbit-module.log"};

    bool verbose{false};
    bool count_warp_level{true};
    bool exclude_pred_off{false};
    bool mangled{true};
    std::string runtime_config_path;
    std::chrono::seconds runtime_config_polling_interval{10};
    std::string measurements_output_dir;

    void print(std::function<void(std::string line)> linePrinter) const;
};

void initialize();
const Configuration& get();
} //namespace config