#include "ErrorUtil.h"

#include <cstring>
#include <cerrno>

namespace util {
std::string getLastErrno() {
    const auto code{errno};
    return "error code " + std::to_string(code) + " (" + strerror(code) + ")";
}
}