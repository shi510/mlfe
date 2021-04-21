#include <gtest/gtest.h>
#include <mlfe/operators_v2/maxpool2d.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators_v2;
namespace fn = mlfe::functional;

TEST(operator_v2, pool2d_max_k2_s2){
    using T = float;
    constexpr int b = 1;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    constexpr int k1 = 2;
    constexpr int k2 = 2;
    auto x = fn::create_variable({b, ci, hi, wi});
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

    auto y = maxpool2d(x, {k1, k2}, {2, 2});

    auto y_ptr = y.data<T>();

    //y(0, 0, 0) = max(1, 2, 5, 6) = 6
    EXPECT_EQ(y_ptr[0], 6);

    //y(0, 0, 1) = max(3, 4, 7, 8) = 8
    EXPECT_EQ(y_ptr[1], 8);

    //y(0, 1, 0) = max(9, 10, 13, 14) = 14
    EXPECT_EQ(y_ptr[2], 14);

    //y(0, 1, 1) = max(11, 12, 15, 16) = 16
    EXPECT_EQ(y_ptr[3], 16);

    //y(1, 0, 0) = max(1, 5, 2, 6) = 6
    EXPECT_EQ(y_ptr[4], 6);

    //y(1, 0, 1) = max(9, 13, 10, 14) = 14
    EXPECT_EQ(y_ptr[5], 14);

    //y(1, 1, 0) = max(3, 7, 4, 8) = 8
    EXPECT_EQ(y_ptr[6], 8);

    //y(1, 1, 1) = max(11, 15, 12, 16) = 16
    EXPECT_EQ(y_ptr[7], 16);
}

TEST(operator_v2, pool2d_max_k2_s2_grad){
    using T = float;
    constexpr int n = 2;
    constexpr int ci = 2;
    constexpr int hi = 4;
    constexpr int wi = 4;
    auto x = fn::create_variable({n, ci, hi, wi});
    // x =
    //    [  1  2  3  4    [ 16 12 8 4
    //       5  6  7  8      15 11 7 3
    //       9 10 11 12      14 10 6 2
    //      13 14 15 16 ]    13  9 5 1 ]
    for(int n = 0; n < hi * wi; ++n){
        x.mutable_data<T>()[n] = n + 1;
    }

    for(int c = 0; c < wi; ++c){
        for(int r = 0; r < hi; ++r){
            const int offset = hi * wi;
            const int idx = offset + r * wi + c;
            x.mutable_data<T>()[idx] = (wi - c - 1) * hi + (hi - r - 1) + 1;
        }
    }
    auto y = maxpool2d(x, {2, 2}, {2, 2});
    y.backprop_v2();

    auto x_grad_ptr = x.grad_v2().data<T>();

    EXPECT_EQ(x_grad_ptr[0], 0);
    EXPECT_EQ(x_grad_ptr[1], 0);
    EXPECT_EQ(x_grad_ptr[2], 0);
    EXPECT_EQ(x_grad_ptr[3], 0);

    EXPECT_EQ(x_grad_ptr[4], 0);
    EXPECT_EQ(x_grad_ptr[5], 1);
    EXPECT_EQ(x_grad_ptr[6], 0);
    EXPECT_EQ(x_grad_ptr[7], 1);

    EXPECT_EQ(x_grad_ptr[8], 0);
    EXPECT_EQ(x_grad_ptr[9], 0);
    EXPECT_EQ(x_grad_ptr[10], 0);
    EXPECT_EQ(x_grad_ptr[11], 0);

    EXPECT_EQ(x_grad_ptr[12], 0);
    EXPECT_EQ(x_grad_ptr[13], 1);
    EXPECT_EQ(x_grad_ptr[14], 0);
    EXPECT_EQ(x_grad_ptr[15], 1);

    EXPECT_EQ(x_grad_ptr[16 + 0], 1);
    EXPECT_EQ(x_grad_ptr[16 + 1], 0);
    EXPECT_EQ(x_grad_ptr[16 + 2], 1);
    EXPECT_EQ(x_grad_ptr[16 + 3], 0);

    EXPECT_EQ(x_grad_ptr[16 + 4], 0);
    EXPECT_EQ(x_grad_ptr[16 + 5], 0);
    EXPECT_EQ(x_grad_ptr[16 + 6], 0);
    EXPECT_EQ(x_grad_ptr[16 + 7], 0);

    EXPECT_EQ(x_grad_ptr[16 + 8], 1);
    EXPECT_EQ(x_grad_ptr[16 + 9], 0);
    EXPECT_EQ(x_grad_ptr[16 + 10], 1);
    EXPECT_EQ(x_grad_ptr[16 + 11], 0);

    EXPECT_EQ(x_grad_ptr[16 + 12], 0);
    EXPECT_EQ(x_grad_ptr[16 + 13], 0);
    EXPECT_EQ(x_grad_ptr[16 + 14], 0);
    EXPECT_EQ(x_grad_ptr[16 + 15], 0);
}