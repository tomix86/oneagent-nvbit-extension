#pragma once

#define STRINGIZE_HELPER(val) #val
#define NAME_OF(function) STRINGIZE_HELPER(function)

#define INSTRUMENTATION__INSTRUCTIONS_COUNT instructionsCount

#define INSTRUMENTATION__OCCUPANCY occupancy