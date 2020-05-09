#pragma once

#include <string>

namespace communication {

std::string nameFromId(int instrumentationFunctionId);
int idFromName(const std::string& instrumentationFunctionName);
bool isIdValid(int instrumentationFunctionId);

} // namespace communication