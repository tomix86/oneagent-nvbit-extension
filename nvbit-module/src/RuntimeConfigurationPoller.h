#pragma once

#include "RuntimeConfiguration.h"

#include <thread>
#include <atomic>
#include <chrono>

namespace config {

class RuntimeConfigurationPoller {
public:
    void initialize(std::string filePath, std::chrono::seconds pollingInterval);

    ~RuntimeConfigurationPoller();

    const RuntimeConfiguration& getConfig();

private:
    std::thread pollerThread;
    RuntimeConfiguration config;
    std::atomic_bool active{false};
};

} // namespace config