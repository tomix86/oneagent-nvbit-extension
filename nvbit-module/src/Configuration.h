#pragma once

#include <filesystem>
#include <functional>
#include <string_view>

namespace config {
struct Configuration {
    std::filesystem::path confFile;
    std::filesystem::path logFile;

    void print(std::function<void(std::string_view line)> linePrinter) const;
};

const Configuration& get();
} //namespace config