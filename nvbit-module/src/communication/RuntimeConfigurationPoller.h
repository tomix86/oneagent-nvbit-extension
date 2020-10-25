#pragma once

#include "RuntimeConfiguration.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

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