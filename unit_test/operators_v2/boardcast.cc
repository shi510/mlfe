#include <gtest/gtest.h>
#include <mlfe/operators/broadcast.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator_v2, broadcast_one_by_n){
    using namespace mlfe;
    namespace fn = functional;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    auto random = [&rng, &dist](){
        return dist(rng);
    };
    auto test_4d_broadcasting = [&](std::vector<int> x_shape,
                                    std::vector<int> target_shape
                                   ){
        auto x = fn::create_variable(x_shape);
        std::generate(x.begin<float>(), x.end<float>(), random);
        auto y = broadcast(x, target_shape);

        EXPECT_EQ(y.dims(), target_shape.size());
        EXPECT_EQ(y.shape()[0], target_shape[0]);
        EXPECT_EQ(y.shape()[1], target_shape[1]);
        EXPECT_EQ(y.shape()[2], target_shape[2]);
        EXPECT_EQ(y.shape()[3], target_shape[3]);

        for(int i = 0; i < y.dim(0); ++i){
            for(int j = 0; j < y.dim(1); ++j){
                for(int k = 0; k < y.dim(2); ++k){
                    for(int l = 0; l < y.dim(3); ++l){
                        auto x_ptr = x.data<float>();
                        auto y_ptr = y.data<float>();
                        auto x_idx = i % x.dim(0) * x.dim(1) * x.dim(2) * x.dim(3) +
                                     j % x.dim(1) * x.dim(2) * x.dim(3) +
                                     k % x.dim(2) * x.dim(3) +
                                     l % x.dim(3);
                        auto y_idx = i * y.dim(1) * y.dim(2) * y.dim(3) +
                                     j * y.dim(2) * y.dim(3) +
                                     k * y.dim(3) +
                                     l ;
                        EXPECT_EQ(x_ptr[x_idx], y_ptr[y_idx]);
                    }
                }
            }
        }
    };
    test_4d_broadcasting({5, 1,  1, 1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 7,  1, 1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 1, 11, 1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 1, 1, 13}, {5, 7, 11 ,13});

    test_4d_broadcasting({5, 7,  1,  1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 7, 11,  1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 1, 11, 13}, {5, 7, 11 ,13});

    test_4d_broadcasting({5, 7, 11,  1}, {5, 7, 11 ,13});
    test_4d_broadcasting({1, 7, 11, 13}, {5, 7, 11 ,13});

    auto test_3d_broadcasting = [&](std::vector<int> x_shape,
                                    std::vector<int> target_shape
                                   ){
        auto x = fn::create_variable(x_shape);
        std::generate(x.begin<float>(), x.end<float>(), random);
        auto y = broadcast(x, target_shape);

        EXPECT_EQ(y.dims(), target_shape.size());
        EXPECT_EQ(y.shape()[0], target_shape[0]);
        EXPECT_EQ(y.shape()[1], target_shape[1]);
        EXPECT_EQ(y.shape()[2], target_shape[2]);

        for(int i = 0; i < y.dim(0); ++i){
            for(int j = 0; j < y.dim(1); ++j){
                for(int k = 0; k < y.dim(2); ++k){
                    auto x_ptr = x.data<float>();
                    auto y_ptr = y.data<float>();
                    auto x_idx = i % x.dim(0) * x.dim(1) * x.dim(2) +
                                 j % x.dim(1) * x.dim(2) +
                                 k % x.dim(2);
                    auto y_idx = i * y.dim(1) * y.dim(2) +
                                 j * y.dim(2) +
                                 k;
                    EXPECT_EQ(x_ptr[x_idx], y_ptr[y_idx]);
                }
            }
        }
    };

    test_3d_broadcasting({5, 1,  1}, {5, 7, 11});
    test_3d_broadcasting({1, 7,  1}, {5, 7, 11});
    test_3d_broadcasting({1, 1, 11}, {5, 7, 11});

    test_3d_broadcasting({5, 7,  1}, {5, 7, 11});
    test_3d_broadcasting({1, 7, 11}, {5, 7, 11});

    auto test_2d_broadcasting = [&](std::vector<int> x_shape,
                                    std::vector<int> target_shape
                                   ){
        auto x = fn::create_variable(x_shape);
        std::generate(x.begin<float>(), x.end<float>(), random);
        auto y = broadcast(x, target_shape);

        EXPECT_EQ(y.dims(), target_shape.size());
        EXPECT_EQ(y.shape()[0], target_shape[0]);
        EXPECT_EQ(y.shape()[1], target_shape[1]);

        for(int i = 0; i < y.dim(0); ++i){
            for(int j = 0; j < y.dim(1); ++j){
                auto x_ptr = x.data<float>();
                auto y_ptr = y.data<float>();
                auto x_idx = i % x.dim(0) * x.dim(1) +
                             j % x.dim(1);
                auto y_idx = i * y.dim(1) +
                             j;
                EXPECT_EQ(x_ptr[x_idx], y_ptr[y_idx]);
            }
        }
    };

    test_2d_broadcasting({5, 1}, {5, 7});
    test_2d_broadcasting({1, 7}, {5, 7});

    auto test_1d_broadcasting = [&](std::vector<int> x_shape,
                                    std::vector<int> target_shape
                                   ){
        auto x = fn::create_variable(x_shape);
        std::generate(x.begin<float>(), x.end<float>(), random);
        auto y = broadcast(x, target_shape);

        EXPECT_EQ(y.dims(), target_shape.size());
        EXPECT_EQ(y.shape()[0], target_shape[0]);

        for(int i = 0; i < y.dim(0); ++i){
            auto x_ptr = x.data<float>();
            auto y_ptr = y.data<float>();
            auto x_idx = i % x.dim(0);
            auto y_idx = i;
            EXPECT_EQ(x_ptr[x_idx], y_ptr[y_idx]);
        }
    };

    test_1d_broadcasting({1}, {5});
}

TEST(operator_v2, broadcast_one_by_n_grad){
    using namespace mlfe;
    using T = float;
    constexpr T grad_eps = 1e-3;
    constexpr T pass_eps = 1e-2;
    namespace fn = functional;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    auto random = [&rng, &dist](){
        return dist(rng);
    };
    auto test_broadcasting_grad = [&](std::vector<int> x_shape,
                                      std::vector<int> target_shape
                                     ){
        Tensor x = fn::create_variable(x_shape);
        std::generate(x.begin<T>(), x.end<T>(), random);
        Tensor y = broadcast(x, target_shape);
        y.backprop_v2();

        EXPECT_EQ(y.dims(), target_shape.size());
        for(int n = 0; n < y.dims(); ++n){
            EXPECT_EQ(y.shape()[n], target_shape[n]);
        }
        auto func = [target_shape](mlfe::Tensor& x){
            return broadcast(x, target_shape);
        };
        auto numerical = numerical_gradient_v2(func, x, grad_eps);
        auto grad_diff = calculate_gradient_diff<T>(numerical, x.grad_v2());
        EXPECT_NEAR(grad_diff, T(0), pass_eps);
    };

    // 4 dims.
    test_broadcasting_grad({2, 1, 1, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 3, 1, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 1, 4, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 1, 1, 5}, {2, 3, 4, 5});

    test_broadcasting_grad({2, 3, 1, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 3, 4, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 1, 4, 5}, {2, 3, 4, 5});

    test_broadcasting_grad({2, 3, 4, 1}, {2, 3, 4, 5});
    test_broadcasting_grad({1, 3, 4, 5}, {2, 3, 4, 5});

    // 3 dims.
    test_broadcasting_grad({2, 1, 1}, {2, 3, 4});
    test_broadcasting_grad({1, 3, 1}, {2, 3, 4});
    test_broadcasting_grad({1, 1, 4}, {2, 3, 4});

    test_broadcasting_grad({2, 3, 1}, {2, 3, 4});
    test_broadcasting_grad({1, 3, 4}, {2, 3, 4});

    // 2 dims.
    test_broadcasting_grad({1, 7}, {5, 7});
    test_broadcasting_grad({5, 1}, {5, 7});

    // 1 dims.
    test_broadcasting_grad({1}, {5});
}

TEST(operator_v2, broadcast_shape_check){

    // matrix vector
    {
        auto a = fn::create_variable({5, 10});
        auto b = broadcast(a, {10});
        EXPECT_EQ(b.shape().size(), 2);
        EXPECT_EQ(b.shape()[0], 5);
        EXPECT_EQ(b.shape()[1], 10);
    }

    // tensor vector
    {
        auto a = fn::create_variable({5, 7, 7, 3});
        auto b = broadcast(a, {3});
        EXPECT_EQ(b.shape().size(), 4);
        EXPECT_EQ(b.shape()[0], 5);
        EXPECT_EQ(b.shape()[1], 7);
        EXPECT_EQ(b.shape()[2], 7);
        EXPECT_EQ(b.shape()[3], 3);
    }
}