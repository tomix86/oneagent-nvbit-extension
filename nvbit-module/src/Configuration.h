#pragma once

#include <cstdint>
#include <functional>

namespace config {
struct Configuration {
    std::string confFile{"nvbit-module.conf"};
    std::string logFile{"nvbit-module.log"};

    // We get some settings that are going to be use to selectively
    // instrument (within a interval of kernel indexes and instructions). By
    // default we instrument everything.
    uint32_t instr_begin_interval{0};
    uint32_t instr_end_interval{UINT32_MAX};
    uint32_t start_grid_num{0};
    uint32_t end_grid_num{UINT32_MAX};
    bool verbose{false};
    bool count_warp_level{true};
    bool exclude_pred_off{false};
    bool active_from_start{true};
    bool mangled{true};
    std::string runtime_config_path{""};
    uint32_t runtime_config_polling_interval{10}; //TODO: make this std::chrono::seconds

    void print(std::function<void(std::string line)> linePrinter) const;
};

void initialize();
const Configuration& get();
} //namespace config