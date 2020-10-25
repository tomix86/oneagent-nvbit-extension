#include "MeasurementsPublisher.h"

#include "Logger.h"
#include "util/ErrorUtil.h"
#include "util/util.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

using std::chrono::system_clock;

namespace communication {

void MeasurementsPublisher::initialize(std::filesystem::path outputDir) {
	if (!std::filesystem::exists(outputDir)) {
		logging::error("Measurements output directory {} does not exist", outputDir.string());
		return;
	}

	this->outputDir = std::move(outputDir);

	logging::info("Measurements will be published to {}", this->outputDir.string());
}

static std::string getFileName() {
	auto timestamp{system_clock::to_time_t(system_clock::now())};
	std::ostringstream ss;
	ss << getpid() << '-' << std::hex << std::this_thread::get_id() << '-' << std::put_time(std::localtime(&timestamp), "%T");
	return ss.str();
}

void MeasurementsPublisher::publish(InstrumentationId instrumentationFunctionId, const std::string& result) {
	if (outputDir.empty()) {
		return;
	}

	const auto outputFileName{outputDir / getFileName()};

	// TODO: atomic save
	std::ofstream output{outputFileName, std::ios::app};
	if (!output) {
		logging::error("Failed to open output file {}: {}", outputFileName.string(), util::getLastErrno());
	}

	output << util::to_underlying_type(instrumentationFunctionId) << ":" << result << std::endl;
	logging::info("Published results ({}: {}) to {}", to_string(instrumentationFunctionId), result, outputFileName.string());
}

} // namespace communication