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

// TODO : add more kernel, stride and padding size test.
// kernel size = 3 x 3
// stride size = 1 x 1
// padding size = 0 x 0
TEST(binary_op, conv2d_k3_s1_p0){
    using T = float;
    constexpr T eps = 1e-5;
    constexpr int n = 2;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int co = 1;
    constexpr int k1 = 3;
    constexpr int k2 = 3;
    auto x = fn::create_variable({n, ci, hi, wi});
    auto w = fn::create_variable({co, ci, k1, k2});
    auto y = fn::conv2d(x, w, {1, 1}, {0, 0});
    
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int n = 0; n < hi * wi; ++n){
        x.mutable_data<T>()[n] = n + 1;
    }
    
    for(int c = 0; c < wi; ++c){
        for(int r = 0; r < hi; ++r){
            const int offset = hi * wi;
            const int idx = offset + r * wi + c;
            x.mutable_data<T>()[idx] = c * hi + r + 1;
        }
    }
    
    // w =
    //    [ 0.1 0.2 0.3    [ 0.1 0.4 0.7
    //      0.4 0.5 0.6      0.2 0.5 0.8
    //      0.7 0.8 0.9 ]    0.3 0.6 0.9 ]
    for(int n = 0; n < k1 * k2; ++n){
        w.mutable_data<T>()[n] = (n + 1) * T(1e-1);
    }
    
    for(int c = 0; c < k2; ++c){
        for(int r = 0; r < k1; ++r){
            const int offset = k1 * k2;
            const int idx = offset + r * k2 + c;
            w.mutable_data<T>()[idx] = (c * k1 + r + 1) * T(1e-1);
        }
    }
    
    y.eval();
    
    //y(0, 0) = 1 * 0.1 +  2 * 0.2 +  3 * 0.3 +
    //          5 * 0.4 +  6 * 0.5 +  7 * 0.6 +
    //          9 * 0.7 + 10 * 0.8 + 11 * 0.9 +
    //
    //          1 * 0.1 + 5 * 0.4 +  9 * 0.7 +
    //          2 * 0.2 + 6 * 0.5 + 10 * 0.8 +
    //          3 * 0.3 + 7 * 0.6 + 11 * 0.9
    //        = 34.8 + 34.8
    EXPECT_LE(y.data<float>()[0], 69.6 + eps);
    EXPECT_GE(y.data<float>()[0], 69.6 - eps);
    
    //y(0, 1) =  2 * 0.1 +  3 * 0.2 +  4 * 0.3 +
    //           6 * 0.4 +  7 * 0.5 +  8 * 0.6 +
    //          10 * 0.7 + 11 * 0.8 + 12 * 0.9 +
    //
    //          5 * 0.1 +  9 * 0.4 + 13 * 0.7 +
    //          6 * 0.2 + 10 * 0.5 + 14 * 0.8 +
    //          7 * 0.3 + 11 * 0.6 + 15 * 0.9
    //        = 92.1
    EXPECT_LE(y.data<float>()[1], 92.1 + eps);
    EXPECT_GE(y.data<float>()[1], 92.1 - eps);
    
    //y(1, 0) =  5 * 0.1 +  6 * 0.2 +  7 * 0.3 +
    //           9 * 0.4 + 10 * 0.5 + 11 * 0.6 +
    //          13 * 0.7 + 14 * 0.8 + 15 * 0.9 +
    //
    //          2 * 0.1 + 6 * 0.4 + 10 * 0.7 +
    //          3 * 0.2 + 7 * 0.5 + 11 * 0.8 +
    //          4 * 0.3 + 8 * 0.6 + 12 * 0.9
    //        = 92.1
    EXPECT_LE(y.data<float>()[2], 92.1 + eps);
    EXPECT_GE(y.data<float>()[2], 92.1 - eps);
    
    //y(1, 1) =  6 * 0.1 +  7 * 0.2 +  8 * 0.3 +
    //          10 * 0.4 + 11 * 0.5 + 12 * 0.6 +
    //          14 * 0.7 + 15 * 0.8 + 16 * 0.9 +
    //
    //          6 * 0.1 + 10 * 0.4 + 14 * 0.7 +
    //          7 * 0.2 + 11 * 0.5 + 15 * 0.8 +
    //          8 * 0.3 + 12 * 0.6 + 16 * 0.9
    //        = 114.6
    EXPECT_LE(y.data<float>()[3], 114.6 + eps);
    EXPECT_GE(y.data<float>()[3], 114.6 - eps);
}

// TODO : add more kernel, stride and padding size test.
// kernel size = 2 x 2
// stride size = 2 x 2
// padding size = 0 x 0
TEST(binary_op, pool2d_max_k2_s2_p0){
    using T = float;
    constexpr int n = 2;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int k1 = 2;
    constexpr int k2 = 2;
    auto x = fn::create_variable({n, ci, hi, wi});
    auto y = fn::pool_max(x, {k1, k2}, {2, 2}, {0, 0});
    
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int n = 0; n < hi * wi; ++n){
        x.mutable_data<T>()[n] = n + 1;
    }
    
    for(int c = 0; c < wi; ++c){
        for(int r = 0; r < hi; ++r){
            const int offset = hi * wi;
            const int idx = offset + r * wi + c;
            x.mutable_data<T>()[idx] = c * hi + r + 1;
        }
    }
    
    y.eval();
    
    //y(0, 0, 0) = max(1, 2, 5, 6) = 6
    EXPECT_EQ(y.data<float>()[0], 6);
    
    //y(0, 0, 1) = max(3, 4, 7, 8) = 8
    EXPECT_EQ(y.data<float>()[1], 8);
    
    //y(0, 1, 0) = max(9, 10, 13, 14) = 14
    EXPECT_EQ(y.data<float>()[2], 14);
    
    //y(0, 1, 1) = max(11, 12, 15, 16) = 16
    EXPECT_EQ(y.data<float>()[3], 16);
    
    //y(1, 0, 0) = max(1, 5, 2, 6) = 6
    EXPECT_EQ(y.data<float>()[4], 6);
    
    //y(1, 0, 1) = max(9, 13, 10, 14) = 14
    EXPECT_EQ(y.data<float>()[5], 14);
    
    //y(1, 1, 0) = max(3, 7, 4, 8) = 8
    EXPECT_EQ(y.data<float>()[6], 8);
    
    //y(1, 1, 1) = max(11, 15, 12, 16) = 16
    EXPECT_EQ(y.data<float>()[7], 16);
}
