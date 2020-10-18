#pragma once

#include <spdlog/spdlog.h>
#include <filesystem>

namespace logging {
void default_initialize();

void initialize(const std::filesystem::path& logFilePath);

template <typename... Ts>
void error(Ts... args) {
    spdlog::error(args...);
}

template <typename... Ts>
void warning(Ts... args) {
    spdlog::warn(args...);
}

template <typename... Ts>
void info(Ts... args) {
    spdlog::info(args...);
}

template <typename... Ts>
void debug(Ts... args) {
    spdlog::debug(args...);
}

}