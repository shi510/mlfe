#include <gtest/gtest.h>
#include "test_fc.hpp"
#include "test_softmax_xent.hpp"
#include "test_simpledb.hpp"
#include "test_conv.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
