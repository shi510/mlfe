#include <gtest/gtest.h>
#include <mlfe/operators_v2/maxpool2d.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators_v2;
namespace fn = mlfe::functional;

TEST(operator_v2, maxpool2d_k2_s2){
    using T = float;
    constexpr int B = 1;
    constexpr int C = 2;
    constexpr int H = 4;
    constexpr int W = 4;
    auto x = fn::create_variable({B, H, W, C});
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int h = 0; h < H; ++h){
        for(int w = 0; w < W; ++w){
            int idx = h * W * C + w * C + 0;
            x.mutable_data<T>()[idx] = h * W + w + 1;
        }
    }


    for(int w = 0; w < W; ++w){
        for(int h = 0; h < H; ++h){
            int idx = h * W * C + w * C + 1;
            x.mutable_data<T>()[idx] = w * H + h + 1;
        }
    }


    auto y = maxpool2d(x, {2, 2}, {2, 2});

    auto y_ptr = y.data<T>();

    //y(0, 0, 0) = max(1, 2, 5, 6) = 6
    EXPECT_EQ(y_ptr[0], 6);

    //y(0, 0, 1) = max(1, 5, 2, 6) = 6
    EXPECT_EQ(y_ptr[1], 6);

    //y(0, 1, 0) = max(3, 4, 7, 8) = 8
    EXPECT_EQ(y_ptr[2], 8);

    //y(0, 1, 1) = max(9, 10, 13, 14) = 14
    EXPECT_EQ(y_ptr[3], 14);

    //y(1, 0, 0) = max(9, 13, 10, 14) = 14
    EXPECT_EQ(y_ptr[4], 14);

    //y(1, 0, 1) = max(3, 7, 4, 8) = 8
    EXPECT_EQ(y_ptr[5], 8);

    //y(1, 1, 0) = max(11, 12, 15, 16) = 16
    EXPECT_EQ(y_ptr[6], 16);

    //y(1, 1, 1) = max(11, 15, 12, 16) = 16
    EXPECT_EQ(y_ptr[7], 16);
}

TEST(operator_v2, maxpool2d_grad_manual_k2_s2){
    using T = float;
    constexpr int B = 1;
    constexpr int C = 2;
    constexpr int H = 4;
    constexpr int W = 4;
    auto x = fn::create_variable({B, H, W, C});
    // x =
    //    [  1  2  3  4    [  1  5  9 13
    //       5  6  7  8       2  6 10 14
    //       9 10 11 12       3  7 11 15
    //      13 14 15 16 ]     4  8 12 16 ]
    for(int h = 0; h < H; ++h){
        for(int w = 0; w < W; ++w){
            int idx = h * W * C + w * C + 0;
            x.mutable_data<T>()[idx] = h * W + w + 1;
        }
    }


    for(int w = 0; w < W; ++w){
        for(int h = 0; h < H; ++h){
            int idx = h * W * C + w * C + 1;
            x.mutable_data<T>()[idx] = w * H + h + 1;
        }
    }

    auto y = maxpool2d(x, {2, 2}, {2, 2});
    y.backprop_v2();

    auto dx_ptr = x.grad_v2().data<T>();
    EXPECT_EQ(dx_ptr[0], 0);
    EXPECT_EQ(dx_ptr[1], 0);
    EXPECT_EQ(dx_ptr[2], 0);
    EXPECT_EQ(dx_ptr[3], 0);

    EXPECT_EQ(dx_ptr[4], 0);
    EXPECT_EQ(dx_ptr[5], 0);
    EXPECT_EQ(dx_ptr[6], 0);
    EXPECT_EQ(dx_ptr[7], 0);

    EXPECT_EQ(dx_ptr[8], 0);
    EXPECT_EQ(dx_ptr[9], 0);
    EXPECT_EQ(dx_ptr[10], 1);
    EXPECT_EQ(dx_ptr[11], 1);

    EXPECT_EQ(dx_ptr[12], 0);
    EXPECT_EQ(dx_ptr[13], 0);
    EXPECT_EQ(dx_ptr[14], 1);
    EXPECT_EQ(dx_ptr[15], 1);

    EXPECT_EQ(dx_ptr[16 + 0], 0);
    EXPECT_EQ(dx_ptr[16 + 1], 0);
    EXPECT_EQ(dx_ptr[16 + 2], 0);
    EXPECT_EQ(dx_ptr[16 + 3], 0);

    EXPECT_EQ(dx_ptr[16 + 4], 0);
    EXPECT_EQ(dx_ptr[16 + 5], 0);
    EXPECT_EQ(dx_ptr[16 + 6], 0);
    EXPECT_EQ(dx_ptr[16 + 7], 0);

    EXPECT_EQ(dx_ptr[16 + 8], 0);
    EXPECT_EQ(dx_ptr[16 + 9], 0);
    EXPECT_EQ(dx_ptr[16 + 10], 1);
    EXPECT_EQ(dx_ptr[16 + 11], 1);

    EXPECT_EQ(dx_ptr[16 + 12], 0);
    EXPECT_EQ(dx_ptr[16 + 13], 0);
    EXPECT_EQ(dx_ptr[16 + 14], 1);
    EXPECT_EQ(dx_ptr[16 + 15], 1);
}