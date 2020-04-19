#pragma once

#include <filesystem>
#include <functional>
#include <string_view>

namespace config {
struct Configuration {
    std::filesystem::path confFile{"nvbit-module.conf"};
    std::filesystem::path logFile{"nvbit-module.log"};

    void print(std::function<void(std::string_view line)> linePrinter) const;
};

void initialize();
const Configuration& get();
} //namespace config