#pragma once

#include "RuntimeConfiguration.h"

#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem>

namespace communication {

class RuntimeConfigurationPoller {
public:
    void initialize(std::filesystem::path filePath, std::chrono::seconds pollingInterval);

    ~RuntimeConfigurationPoller();

    const RuntimeConfiguration& getConfig();

private:
    std::thread pollerThread;
    RuntimeConfiguration config;
    std::atomic_bool active{false};
};

} // namespace communication