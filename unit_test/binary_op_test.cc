#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/operators.h>
#include <cmath>

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

TEST(binary_op, matmul){

    // [2, 2] = [2, 3] x [3, 2]
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({3, 2});

        a.mutable_data<float>()[0] = 1;
        a.mutable_data<float>()[1] = 2;
        a.mutable_data<float>()[2] = 3;

        a.mutable_data<float>()[3] = 4;
        a.mutable_data<float>()[4] = 5;
        a.mutable_data<float>()[5] = 6;

        b.mutable_data<float>()[0] = 10;
        b.mutable_data<float>()[1] = 20;

        b.mutable_data<float>()[2] = 30;
        b.mutable_data<float>()[3] = 40;

        b.mutable_data<float>()[4] = 50;
        b.mutable_data<float>()[5] = 60;

        auto c = fn::matmul(a, b);
        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);
        c.eval();

        EXPECT_EQ(c.data<float>()[0], 10 + 60 + 150);
        EXPECT_EQ(c.data<float>()[1], 20 + 80 + 180);

        EXPECT_EQ(c.data<float>()[2], 40 + 150 + 300);
        EXPECT_EQ(c.data<float>()[3], 80 + 200 + 360);
    }
    
    // [2, 2] = [2, 3] x [2, 3]^T
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({2, 3});

        a.mutable_data<float>()[0] = 1;
        a.mutable_data<float>()[1] = 2;
        a.mutable_data<float>()[2] = 3;

        a.mutable_data<float>()[3] = 4;
        a.mutable_data<float>()[4] = 5;
        a.mutable_data<float>()[5] = 6;

        b.mutable_data<float>()[0] = 10;
        b.mutable_data<float>()[1] = 20;
        b.mutable_data<float>()[2] = 30;

        b.mutable_data<float>()[3] = 40;
        b.mutable_data<float>()[4] = 50;
        b.mutable_data<float>()[5] = 60;

        auto c = fn::matmul(a, b, false, true);
        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);
        c.eval();

        EXPECT_EQ(c.data<float>()[0], 10 + 40 + 90);
        EXPECT_EQ(c.data<float>()[1], 40 + 100 + 180);

        EXPECT_EQ(c.data<float>()[2], 40 + 100 + 180);
        EXPECT_EQ(c.data<float>()[3], 160 + 250 + 360);
    }

    // [3, 3] = [2, 3]^T x [2, 3]
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({2, 3});

        a.mutable_data<float>()[0] = 1;
        a.mutable_data<float>()[1] = 2;
        a.mutable_data<float>()[2] = 3;

        a.mutable_data<float>()[3] = 4;
        a.mutable_data<float>()[4] = 5;
        a.mutable_data<float>()[5] = 6;

        b.mutable_data<float>()[0] = 10;
        b.mutable_data<float>()[1] = 20;
        b.mutable_data<float>()[2] = 30;

        b.mutable_data<float>()[3] = 40;
        b.mutable_data<float>()[4] = 50;
        b.mutable_data<float>()[5] = 60;

        auto c = fn::matmul(a, b, true);
        EXPECT_EQ(c.size(), 9);
        EXPECT_EQ(c.shape()[0], 3);
        EXPECT_EQ(c.shape()[1], 3);
        c.eval();

        EXPECT_EQ(c.data<float>()[0], 10 + 160);
        EXPECT_EQ(c.data<float>()[1], 20 + 200);
        EXPECT_EQ(c.data<float>()[2], 30 + 240);

        EXPECT_EQ(c.data<float>()[3], 20 + 200);
        EXPECT_EQ(c.data<float>()[4], 40 + 250);
        EXPECT_EQ(c.data<float>()[5], 60 + 300);
        
        EXPECT_EQ(c.data<float>()[6], 30 + 240);
        EXPECT_EQ(c.data<float>()[7], 60 + 300);
        EXPECT_EQ(c.data<float>()[8], 90 + 360);
    }

    // [2, 2] = [3, 2]^T x [2, 3]^T
    {
        auto a = fn::create_variable({3, 2});
        auto b = fn::create_variable({2, 3});

        a.mutable_data<float>()[0] = 1;
        a.mutable_data<float>()[1] = 2;

        a.mutable_data<float>()[2] = 3;
        a.mutable_data<float>()[3] = 4;

        a.mutable_data<float>()[4] = 5;
        a.mutable_data<float>()[5] = 6;

        b.mutable_data<float>()[0] = 10;
        b.mutable_data<float>()[1] = 20;
        b.mutable_data<float>()[2] = 30;

        b.mutable_data<float>()[3] = 40;
        b.mutable_data<float>()[4] = 50;
        b.mutable_data<float>()[5] = 60;

        auto c = fn::matmul(a, b, true, true);
        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);
        c.eval();

        EXPECT_EQ(c.data<float>()[0], 10 + 60 + 150);
        EXPECT_EQ(c.data<float>()[1], 40 + 150 + 300);

        EXPECT_EQ(c.data<float>()[2], 20 + 80 + 180);
        EXPECT_EQ(c.data<float>()[3], 80 + 200 + 360);
    }
}

TEST(binary_op, squared_difference){
    auto a = fn::create_variable({2, 2});
    auto b = fn::create_variable({2, 2});

    a.mutable_data<float>()[0] = 1;
    a.mutable_data<float>()[1] = 2;
    a.mutable_data<float>()[2] = 3;
    a.mutable_data<float>()[3] = 4;

    b.mutable_data<float>()[0] = -10;
    b.mutable_data<float>()[1] = 20;
    b.mutable_data<float>()[2] = -30;
    b.mutable_data<float>()[3] = 40;

    auto c = fn::squared_difference(a, b);
    EXPECT_EQ(c.size(), 4);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    c.eval();

    EXPECT_EQ(c.data<float>()[0], std::pow(1 - -10, 2));
    EXPECT_EQ(c.data<float>()[1], std::pow(2 - 20, 2));

    EXPECT_EQ(c.data<float>()[2], std::pow(3 - -30, 2));
    EXPECT_EQ(c.data<float>()[3], std::pow(4 - 40, 2));
}