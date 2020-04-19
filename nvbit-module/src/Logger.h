#pragma once

#include <spdlog/spdlog.h>
#include <string>

namespace logging {
void initialize(const std::string& logFilePath);

template <typename... Ts>
void info(Ts... args) {
    spdlog::info(args...);
}

template <typename... Ts>
void debug(Ts... args) {
    spdlog::debug(args...);
}

}