#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>

using namespace mlfe;
namespace fn = functional;

TEST(binary_op, add){
    using T = float;
    auto x1 = fn::create_variable({2, 2});
    auto x2 = fn::create_variable({2, 2});
    auto result = fn::add(x1, x2);
    x1.mutable_data<T>()[0] = 1;
    x1.mutable_data<T>()[1] = 1;
    x1.mutable_data<T>()[2] = 1;
    x1.mutable_data<T>()[3] = 1;

    x2.mutable_data<T>()[0] = -1;
    x2.mutable_data<T>()[1] = -3;
    x2.mutable_data<T>()[2] = 5;
    x2.mutable_data<T>()[3] = 7;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 0);
    EXPECT_EQ(result.data<T>()[1], -2);
    EXPECT_EQ(result.data<T>()[2], 6);
    EXPECT_EQ(result.data<T>()[3], 8);
}

TEST(binary_op, sub){
    using T = float;
    auto x1 = fn::create_variable({2, 2});
    auto x2 = fn::create_variable({2, 2});
    auto result = fn::sub(x1, x2);
    x1.mutable_data<T>()[0] = 1;
    x1.mutable_data<T>()[1] = 1;
    x1.mutable_data<T>()[2] = 1;
    x1.mutable_data<T>()[3] = 1;

    x2.mutable_data<T>()[0] = -1;
    x2.mutable_data<T>()[1] = -3;
    x2.mutable_data<T>()[2] = 5;
    x2.mutable_data<T>()[3] = 7;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 2);
    EXPECT_EQ(result.data<T>()[1], 4);
    EXPECT_EQ(result.data<T>()[2], -4);
    EXPECT_EQ(result.data<T>()[3], -6);
}

TEST(binary_op, mul){
    using T = float;
    auto x1 = fn::create_variable({2, 2});
    auto x2 = fn::create_variable({2, 2});
    auto result = fn::mul(x1, x2);
    x1.mutable_data<T>()[0] = 2;
    x1.mutable_data<T>()[1] = -2;
    x1.mutable_data<T>()[2] = 2;
    x1.mutable_data<T>()[3] = -2;

    x2.mutable_data<T>()[0] = -1;
    x2.mutable_data<T>()[1] = -3;
    x2.mutable_data<T>()[2] = 5;
    x2.mutable_data<T>()[3] = 7;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], -2);
    EXPECT_EQ(result.data<T>()[1], 6);
    EXPECT_EQ(result.data<T>()[2], 10);
    EXPECT_EQ(result.data<T>()[3], -14);
}

TEST(binary_op, div){
    using T = float;
    auto x1 = fn::create_variable({2, 2});
    auto x2 = fn::create_variable({2, 2});
    auto result = fn::div(x1, x2);
    x1.mutable_data<T>()[0] = 10;
    x1.mutable_data<T>()[1] = 10;
    x1.mutable_data<T>()[2] = 10;
    x1.mutable_data<T>()[3] = 10;

    x2.mutable_data<T>()[0] = 0;
    x2.mutable_data<T>()[1] = -3;
    x2.mutable_data<T>()[2] = 5;
    x2.mutable_data<T>()[3] = -7;
    result.eval();
    EXPECT_EQ(std::isinf(result.data<T>()[0]), true);

    EXPECT_LE(result.data<T>()[1], -3.3333 + 1e-4);
    EXPECT_GE(result.data<T>()[1], -3.3333 - 1e-4);

    EXPECT_LE(result.data<T>()[2], 2 + 1e-4);
    EXPECT_GE(result.data<T>()[2], 2 - 1e-4);

    EXPECT_LE(result.data<T>()[3], -1.4285 + 1e-4);
    EXPECT_GE(result.data<T>()[3], -1.4285 - 1e-4);
}