#include "util/cuda_utilities.h"

#include <gtest/gtest.h>

namespace util {

TEST(ComputeCapabilityTest, comparison_operators) {
	ASSERT_EQ((ComputeCapability{1, 1}), (ComputeCapability{1, 1}));
	ASSERT_NE((ComputeCapability{1, 0}), (ComputeCapability{1, 1}));
	ASSERT_LT((ComputeCapability{1, 0}), (ComputeCapability{1, 1}));
	ASSERT_LE((ComputeCapability{1, 0}), (ComputeCapability{1, 1}));
	ASSERT_LE((ComputeCapability{1, 1}), (ComputeCapability{1, 1}));
	ASSERT_GT((ComputeCapability{1, 1}), (ComputeCapability{1, 0}));
	ASSERT_GE((ComputeCapability{1, 1}), (ComputeCapability{1, 0}));
	ASSERT_GE((ComputeCapability{1, 1}), (ComputeCapability{1, 1}));
}

TEST(ComputeCapabilityTest, toString) {
	ASSERT_EQ("1,1", (ComputeCapability{1, 1}).toString());
	ASSERT_EQ("2,0", (ComputeCapability{2, 0}).toString());
}

} // namespace util