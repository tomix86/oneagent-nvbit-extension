#include "Logger.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

using namespace std::string_literals;

namespace logging {

void default_initialize() try {
    spdlog::default_logger()->set_level(spdlog::level::off);
} catch (...){
}

void initialize(const std::filesystem::path& logFilePath, bool enableConsoleLog) try {
	auto file_sink{std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath)};
	file_sink->set_level(spdlog::level::trace);

	std::vector<spdlog::sink_ptr> sinks{file_sink};
	if(enableConsoleLog) {
		auto& console_sink{sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>())};
		console_sink->set_level(spdlog::level::trace);
		console_sink->set_pattern("[%^%l%$] [NVBit] %v");
	}

	const auto logger{ std::make_shared<spdlog::logger>("log", sinks.cbegin(), sinks.cend()) };
	logger->set_level(spdlog::level::debug);
	logger->flush_on(spdlog::level::debug);
	spdlog::set_default_logger(logger);

	spdlog::info("Logger initialized");
	spdlog::debug("Current working directory: \"{}\", pid {}", std::filesystem::current_path().string(), getpid());
} catch (const spdlog::spdlog_ex& ex) {
    throw std::runtime_error{"Logger initialization failed: "s + ex.what()};
}

}