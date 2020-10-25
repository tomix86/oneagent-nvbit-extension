#pragma once

#include <type_traits>

namespace util {

template <typename T>
constexpr auto to_underlying_type(T val) {
	return static_cast<std::underlying_type_t<T>>(val);
}

// Test whether val ∈ [low; high)
template <typename T>
constexpr bool is_in_range(const T& val, const T& low, const T& high) {
	return val >= low && val < high;
}

} // namespace util