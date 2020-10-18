#include "communication/InstrumentationId.h"

#include <gtest/gtest.h>

TEST(Communication, InstrumentationId_to_string) {
    ASSERT_EQ("instructions_count", communication::to_string(communication::InstrumentationId::instructions_count));
}
