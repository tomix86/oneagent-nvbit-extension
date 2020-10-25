#include "ErrorUtil.h"

#include <cerrno>
#include <cstring>

namespace util {
std::string getLastErrno() {
	const auto code{errno};
	return "error code " + std::to_string(code) + " (" + strerror(code) + ")";
}
} // namespace util