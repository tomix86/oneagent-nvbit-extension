#include "RuntimeConfigurationPoller.h"

#include "RuntimeConfiguration.h"
#include "Logger.h"

#include <boost/algorithm/string/join.hpp>

using namespace std::chrono_literals;

using myclock = std::chrono::steady_clock;

namespace config {

void RuntimeConfigurationPoller::initialize(std::string filePath, std::chrono::seconds pollingInterval) {
    logging::info("Runtime configuration will be polled from {} every {}s", filePath, pollingInterval.count());

    pollerThread = std::thread([this, filePath = std::move(filePath), pollingInterval](){
        auto begin{myclock::now()};
        
        while(active) {
            const auto now{myclock::now()};
            if(now - begin > pollingInterval) {
                begin = now;

                try {
                    config.load(filePath);
                    logging::debug("Loaded runtime configuration: {}", boost::algorithm::join(config.getInstrumentationFunctions(), " "));
                } catch(const std::exception& ex) {
                    logging::error("Failed to load runtime configuration file: {}", ex.what());
                }
            }

            std::this_thread::sleep_for(10ms);
        }
    });

    active = true;
    logging::info("Runtime configuration poller activated");
}

RuntimeConfigurationPoller::~RuntimeConfigurationPoller() {
    logging::info("Runtime configuration poller deactivated");

    active = false;
    if(pollerThread.joinable()) {
        try {
            pollerThread.join();
        } catch(const std::exception& ex) {
            logging::warning("Failed to join runtime configuration poller thread: {}", ex.what());
        }
    }

    logging::info("Runtime configuration poller finished");

}

const RuntimeConfiguration& RuntimeConfigurationPoller::getConfig() {
    return config;
}

} // namespace config