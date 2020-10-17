#include "MeasurementsPublisher.h"

#include "Logger.h"
#include "ErrorUtil.h"
#include "util.h"

#include <filesystem>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>


using std::chrono::system_clock;

namespace communication {

void MeasurementsPublisher::initialize(std::string outputDir){ 
    if(!std::filesystem::exists(outputDir)) {
        logging::error("Measurements output directory {} does not exist", outputDir);
        return;
    }

    this->outputDir = std::move(outputDir);

    logging::info("Measurements will be published to {}", this->outputDir);
}

static std::string getFileName() {
    auto timestamp{system_clock::to_time_t(system_clock::now())};
    std::ostringstream ss;
    ss << getpid() << '-' << std::hex << std::this_thread::get_id() << '-' << std::put_time(std::localtime(&timestamp), "%T");
    return ss.str();
}

void MeasurementsPublisher::publish(InstrumentationId instrumentationFunctionId, const std::string& result){
    if(outputDir.empty()){
        return;
    }

    const auto outputFileName{std::filesystem::path{outputDir}.lexically_normal() / getFileName()};
    
    //TODO: atomic save
    std::ofstream output{outputFileName, std::ios::app};
    if(!output) {
        logging::error("Failed to open output file {}: {}", outputFileName.string(), util::getLastErrno());
    }

    output << util::to_underlying_type(instrumentationFunctionId) << ":" << result << std::endl;
    logging::info("Published results ({}: {}) to {}", to_string(instrumentationFunctionId), result, outputFileName.string());
}

} // namespace communication