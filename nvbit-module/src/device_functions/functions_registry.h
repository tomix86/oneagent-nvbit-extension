#pragma once

#define STRINGIZE_HELPER(val) #val
#define NAME_OF(function) STRINGIZE_HELPER(function)

//TODO: A short description goes here
#define INSTRUMENTATION__INSTRUCTIONS_COUNT instructionsCount