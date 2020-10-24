#include "Configuration.h"
#include "Logger.h"
 
#include <string_view>
#include <cstdlib>
#include <unistd.h>

template <typename... Ts>
void writeStderr(Ts... parts) {
    const auto printer{[](std::string_view message){
        (void)!write(STDERR_FILENO, message.data(), message.size());
    }};

    (printer(parts), ...);
    (void)!write(STDERR_FILENO, "\n", 1);
}

void __attribute__((constructor)) initialize() try {
    // Disable NVBit banner printing
    setenv("NOBANNER", "1", 1);

    logging::default_initialize();
    logging::initialize(config::get().logFile, config::get().console_log_enabled);
    
    logging::info("NVBit module loaded, configuration:");
    config::get().print([](auto line){
        logging::info("\t{}", line);
    });
} catch (const std::exception& ex) {
    writeStderr("Error in constructor: ", ex.what());
} catch (...) {
    writeStderr("Unknown error in constructor");
}

void __attribute__((destructor)) finalize() try {
    logging::info("NVBit module unloaded");
} catch (const std::exception& ex) {
    writeStderr("Error in destructor: ", ex.what());
} catch (...) {
    writeStderr("Unknown error in destructor");
}