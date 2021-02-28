#include "RuntimeConfiguration.h"

#include "util/ErrorUtil.h"

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <fstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

using namespace std::string_literals;

namespace communication {

// TODO: thread safety
void RuntimeConfiguration::load(const std::filesystem::path& filePath) {
	// TODO: reload only if file contents changed

	std::ifstream input{filePath};
	if (!input) {
		throw std::runtime_error{"opening file failed: " + util::getLastErrno()};
	}

	instrumentationFunctions.clear();

	const auto pid{std::to_string(getpid())};

	std::string buf;
	while (std::getline(input, buf)) {
		std::vector<std::string> tokens;
		boost::split(tokens, buf, boost::is_any_of(":"));
		if (tokens.size() != 2) {
			continue;
		}

		if (tokens.front() != pid) {
			continue;
		}

		try {
			std::vector<std::string> functionIds;
			boost::split(functionIds, tokens.back(), boost::is_any_of(","));
			for (const auto& id : functionIds | boost::adaptors::transformed([](auto i) { return std::stoi(i); })) {
				if (!isInstrumentationIdValid(id)) {
					throw std::runtime_error{"Invalid instrumentation id: " + std::to_string(id)};
				}

				instrumentationFunctions.push_back(InstrumentationId{id});
			}

			if (!isInstrumentationSetValid(instrumentationFunctions)) {
				throw std::runtime_error{"Multiple injection routines cannot be combined together"};
			}
		} catch (const std::exception& ex) {
			throw std::runtime_error{"malformed key encountered: " + tokens.back() + ", " + ex.what()};
		}
	}
}

std::vector<InstrumentationId> RuntimeConfiguration::getInstrumentationFunctions() const {
	return instrumentationFunctions;
}

} // namespace communication
