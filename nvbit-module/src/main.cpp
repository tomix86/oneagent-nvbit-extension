#include "Configuration.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <string_view>
#include <unistd.h>
#include <filesystem>

template <typename... Ts>
static void logError(Ts... parts) {
    const auto printer{[](std::string_view message){
        write(STDERR_FILENO, message.data(), message.size());
    }};

    (printer(parts), ...);
    write(STDERR_FILENO, "\n", 1);
}

static void initLogger() try {
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(config::get().logFile);
	file_sink->set_level(spdlog::level::trace);

	const auto logger{ std::make_shared<spdlog::logger>("log", file_sink) };
	logger->set_level(spdlog::level::debug);
	logger->flush_on(spdlog::level::debug);
	spdlog::set_default_logger(logger);

	spdlog::info("Logger initialized");
	spdlog::debug("Current working directory: \"{}\", pid {}", std::filesystem::current_path().string(), getpid());
    spdlog::debug("Module configuration:");
    config::get().print([](auto line){
        spdlog::debug("\t{}", line);
    });
} catch (const spdlog::spdlog_ex& ex) {
    logError("Logger initialization failed: ", ex.what());
}
 
void __attribute__((constructor)) initialize() try {
    config::initialize();
    initLogger();
} catch (const std::exception& ex) {
    logError("Error in constructor: ", ex.what());
} catch (...) {
    logError("Unknown error in constructor");
}

void __attribute__((destructor)) finalize() try {
} catch (const std::exception& ex) {
    logError("Error in destructor: ", ex.what());
} catch (...) {
    logError("Unknown error in destructor");
}