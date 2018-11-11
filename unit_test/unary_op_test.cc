#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>

using namespace mlfe;
namespace fn = functional;

TEST(unary_op, negative){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::negative(var);
    var.mutable_data<T>()[0] = -1.35;
    var.mutable_data<T>()[1] = 357.5;
    var.mutable_data<T>()[2] = -3.5;
    var.mutable_data<T>()[3] = 3.15;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], -var.data<T>()[0]);
    EXPECT_EQ(result.data<T>()[1], -var.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], -var.data<T>()[2]);
    EXPECT_EQ(result.data<T>()[3], -var.data<T>()[3]);
}

TEST(unary_op, sigmoid){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::sigmoid(var);
    var.mutable_data<T>()[0] = 0;
    var.mutable_data<T>()[1] = -0.5;
    var.mutable_data<T>()[2] = 0.5;
    var.mutable_data<T>()[3] = 1;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 0.5);

    EXPECT_LE(result.data<T>()[1], 0.3775 + 1e-4);
    EXPECT_GE(result.data<T>()[1], 0.3775 - 1e-4);

    EXPECT_LE(result.data<T>()[2], 0.6225 + 1e-4);
    EXPECT_GE(result.data<T>()[2], 0.6225 - 1e-4);

    EXPECT_LE(result.data<T>()[3], 0.7311 + 1e-4);
    EXPECT_GE(result.data<T>()[3], 0.7311 - 1e-4);
}

TEST(unary_op, relu){
    using T = float;
    auto var = fn::create_variable({2, 2});
    auto result = fn::relu(var);
    var.mutable_data<T>()[0] = -1.35;
    var.mutable_data<T>()[1] = 357.5;
    var.mutable_data<T>()[2] = -3.5;
    var.mutable_data<T>()[3] = 3.15;
    result.eval();
    EXPECT_EQ(result.data<T>()[0], 0);
    EXPECT_EQ(result.data<T>()[1], var.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], 0);
    EXPECT_EQ(result.data<T>()[3], var.data<T>()[3]);
}