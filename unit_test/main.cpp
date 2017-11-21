#include <gtest/gtest.h>
#include "test_fc_operator.hpp"
#include "test_softmax_xent.hpp"
#include "test_simpledb.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
