#include "util/util.h"

#include <gtest/gtest.h>

TEST(Util, is_in_range) {
	ASSERT_TRUE(util::is_in_range(1, 0, 2));
}