#include "Logger.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <string>
#include <filesystem>

using namespace std::string_literals;

namespace logging {

void default_initialize() try {
    spdlog::default_logger()->set_level(spdlog::level::off);
} catch (...){
}

void initialize(const std::string& logFilePath) try {
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath);
	file_sink->set_level(spdlog::level::trace);

	const auto logger{ std::make_shared<spdlog::logger>("log", file_sink) };
	logger->set_level(spdlog::level::debug);
	logger->flush_on(spdlog::level::debug);
	spdlog::set_default_logger(logger);

	spdlog::info("Logger initialized");
	spdlog::debug("Current working directory: \"{}\", pid {}", std::filesystem::current_path().string(), getpid());
} catch (const spdlog::spdlog_ex& ex) {
    throw std::runtime_error{"Logger initialization failed: "s + ex.what()};
}

}