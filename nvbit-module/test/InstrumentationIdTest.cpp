#include "communication/InstrumentationId.h"

#include <gtest/gtest.h>

namespace communication {

using Id = InstrumentationId;

TEST(Communication, InstrumentationId_to_string) {
	EXPECT_EQ("instructions_count", to_string(Id::instructions_count));
	EXPECT_EQ("occupancy", to_string(Id::occupancy));
	EXPECT_EQ("gmem_access_coalescence", to_string(Id::gmem_access_coalescence));
	EXPECT_EQ("branch_divergence", to_string(Id::branch_divergence));
	EXPECT_THROW(to_string(static_cast<Id>(42)), std::invalid_argument);
}

TEST(Communication, InstrumentationId_is_instrumentation_id_valid) {
	constexpr auto maxId{3};
	for (int i{0}; i <= maxId; ++i) {
		EXPECT_TRUE(isInstrumentationIdValid(i));
	}

	EXPECT_FALSE(isInstrumentationIdValid(maxId + 1));
}

TEST(Communication, InstrumentationId_is_instrumentation_set_valid) {
	EXPECT_TRUE(isInstrumentationSetValid({Id::occupancy}));
	EXPECT_TRUE(isInstrumentationSetValid({Id::instructions_count}));
	EXPECT_TRUE(isInstrumentationSetValid({Id::instructions_count, Id::occupancy}));
	EXPECT_FALSE(isInstrumentationSetValid({Id::instructions_count, Id::gmem_access_coalescence, Id::occupancy}));
	EXPECT_FALSE(isInstrumentationSetValid({Id::instructions_count, Id::gmem_access_coalescence}));
	EXPECT_FALSE(isInstrumentationSetValid({Id::gmem_access_coalescence, Id::branch_divergence}));
	EXPECT_FALSE(isInstrumentationSetValid({Id::instructions_count, Id::gmem_access_coalescence, Id::branch_divergence}));
}

} // namespace communication