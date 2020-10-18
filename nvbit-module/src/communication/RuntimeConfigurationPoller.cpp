#include "RuntimeConfigurationPoller.h"

#include "RuntimeConfiguration.h"
#include "Logger.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>

using namespace std::chrono_literals;

using std::chrono::steady_clock;
using boost::algorithm::join;
using boost::adaptors::transformed;

namespace communication {

void RuntimeConfigurationPoller::initialize(std::filesystem::path filePath, std::chrono::seconds pollingInterval) {
    logging::info("Runtime configuration will be polled from {} every {}s", filePath.string(), pollingInterval.count());

    pollerThread = std::thread([this, filePath = std::move(filePath), pollingInterval](){
        auto begin{steady_clock::now()};
        
        while(active) {
            const auto now{steady_clock::now()};
            if(now - begin > pollingInterval) {
                begin = now;

                try {
                    config.load(filePath);
                    logging::debug("Loaded runtime configuration: {}", join(config.getInstrumentationFunctions() | transformed(to_string), " "));
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

} // namespace communication