#pragma once

#include <string> 
#include <cstdint>
#include <chrono>
#include <functional>
#include <filesystem>

namespace config {
struct Configuration {
    std::filesystem::path confFile{"nvbit-module.conf"};
    std::filesystem::path logFile{"nvbit-module.log"};
    
    bool verbose{false};
    bool count_warp_level{true};
    bool exclude_pred_off{false};
    bool mangled{true};
    std::filesystem::path runtime_config_path;
    std::chrono::seconds runtime_config_polling_interval{10};
    std::filesystem::path measurements_output_dir;

    void print(std::function<void(std::string line)> linePrinter) const;
};

void initialize();
const Configuration& get();
} //namespace config