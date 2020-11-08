#include "communication/InstrumentationId.h"

#include <gtest/gtest.h>

namespace communication {

TEST(Communication, InstrumentationId_to_string) {
	EXPECT_EQ("instructions_count", to_string(InstrumentationId::instructions_count));
	EXPECT_EQ("occupancy", to_string(InstrumentationId::occupancy));
	EXPECT_EQ("gmem_access_coalescence", to_string(InstrumentationId::gmem_access_coalescence));
	EXPECT_EQ("branch_divergence", to_string(InstrumentationId::branch_divergence));
	EXPECT_THROW(to_string(static_cast<InstrumentationId>(42)), std::invalid_argument);
}

TEST(Communication, InstrumentationId_is_instumentation_id_valid) {
	constexpr auto maxId{3};
	for (int i{0}; i <= maxId; ++i) {
		EXPECT_TRUE(isInstrumentationIdValid(i));
	}

	EXPECT_FALSE(isInstrumentationIdValid(maxId + 1));
}

} // namespace communication