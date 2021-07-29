#include <gtest/gtest.h>
#include <mlfe/core/tensor.h>
#include <mlfe/operators/matmul.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator, matmul){
    using T = float;
    // [2, 2] = [2, 3] x [3, 2]
    {
        auto a = Tensor::from_vector<T>({1, 2, 3, 4, 5, 6}, {2, 3});
        auto b = Tensor::from_vector<T>({10, 20, 30, 40, 50, 60}, {3, 2});
        auto c = matmul(a, b);
        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);

        EXPECT_EQ(c.data<float>()[0], 10 + 60 + 150);
        EXPECT_EQ(c.data<float>()[1], 20 + 80 + 180);

        EXPECT_EQ(c.data<float>()[2], 40 + 150 + 300);
        EXPECT_EQ(c.data<float>()[3], 80 + 200 + 360);
    }

    // [2, 2] = [2, 3] x [2, 3]^T
    {
        auto a = Tensor::from_vector<T>({1, 2, 3, 4, 5, 6}, {2, 3});
        auto b = Tensor::from_vector<T>({10, 20, 30, 40, 50, 60}, {2, 3});
        auto c = matmul(a, b, false, true);

        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);

        EXPECT_EQ(c.data<float>()[0], 10 + 40 + 90);
        EXPECT_EQ(c.data<float>()[1], 40 + 100 + 180);

        EXPECT_EQ(c.data<float>()[2], 40 + 100 + 180);
        EXPECT_EQ(c.data<float>()[3], 160 + 250 + 360);
    }

    // [3, 3] = [2, 3]^T x [2, 3]
    {
        auto a = Tensor::from_vector<T>({1, 2, 3, 4, 5, 6}, {2, 3});
        auto b = Tensor::from_vector<T>({10, 20, 30, 40, 50, 60}, {2, 3});
        auto c = matmul(a, b, true);
        EXPECT_EQ(c.size(), 9);
        EXPECT_EQ(c.shape()[0], 3);
        EXPECT_EQ(c.shape()[1], 3);

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
        auto a = Tensor::from_vector<T>({1, 2, 3, 4, 5, 6}, {3, 2});
        auto b = Tensor::from_vector<T>({10, 20, 30, 40, 50, 60}, {2, 3});
        auto c = matmul(a, b, true, true);
        EXPECT_EQ(c.size(), 4);
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 2);

        EXPECT_EQ(c.data<float>()[0], 10 + 60 + 150);
        EXPECT_EQ(c.data<float>()[1], 40 + 150 + 300);

        EXPECT_EQ(c.data<float>()[2], 20 + 80 + 180);
        EXPECT_EQ(c.data<float>()[3], 80 + 200 + 360);
    }
}

TEST(operator, matmul_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;

    // [2, 2] = [2, 3] x [3, 2]
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({3, 2});
        auto analytical_a = fn::create_variable(a.shape());
        auto analytical_b = fn::create_variable(b.shape());
        std::mt19937 rng;
        std::uniform_real_distribution<T> dist(-1, 1);

        std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        auto c = matmul(a, b);
        c.backprop_v2();
        std::copy(a.grad_v2().begin<T>(), a.grad_v2().end<T>(),
            analytical_a.begin<T>());
        std::copy(b.grad_v2().begin<T>(), b.grad_v2().end<T>(),
            analytical_b.begin<T>());
        auto func1 = [b](mlfe::Tensor & x){
            return matmul(x, b);
        };
        auto func2 = [a](mlfe::Tensor & x){
            return matmul(a, x);
        };

        auto numerical = numerical_gradient_v2(func1, a, grad_eps);
        auto diff = calculate_gradient_diff<T>(numerical, analytical_a);
        EXPECT_NEAR(diff, T(0), pass_eps);

        numerical = numerical_gradient_v2(func2, b, grad_eps);
        diff = calculate_gradient_diff<T>(numerical, analytical_b);
        EXPECT_NEAR(diff, T(0), pass_eps);
    }

    // // [2, 2] = [2, 3] x [2, 3]^T
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({2, 3});
        auto analytical_a = fn::create_variable(a.shape());
        auto analytical_b = fn::create_variable(b.shape());
        std::mt19937 rng;
        std::uniform_real_distribution<T> dist(-1, 1);

        std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        auto c = matmul(a, b, false, true);
        c.backprop_v2();
        std::copy(a.grad_v2().begin<T>(), a.grad_v2().end<T>(),
            analytical_a.begin<T>());
        std::copy(b.grad_v2().begin<T>(), b.grad_v2().end<T>(),
            analytical_b.begin<T>());
        auto func1 = [b](mlfe::Tensor & x){
            return matmul(x, b, false, true);
        };
        auto func2 = [a](mlfe::Tensor & x){
            return matmul(a, x, false, true);
        };

        auto numerical = numerical_gradient_v2(func1, a, grad_eps);
        auto diff = calculate_gradient_diff<T>(numerical, analytical_a);
        EXPECT_NEAR(diff, T(0), pass_eps);

        numerical = numerical_gradient_v2(func2, b, grad_eps);
        diff = calculate_gradient_diff<T>(numerical, analytical_b);
        EXPECT_NEAR(diff, T(0), pass_eps);
    }

    // [3, 3] = [2, 3]^T x [2, 3]
    {
        auto a = fn::create_variable({2, 3});
        auto b = fn::create_variable({2, 3});
        auto analytical_a = fn::create_variable(a.shape());
        auto analytical_b = fn::create_variable(b.shape());
        std::mt19937 rng;
        std::uniform_real_distribution<T> dist(-1, 1);

        std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        auto c = matmul(a, b, true);
        c.backprop_v2();
        std::copy(a.grad_v2().begin<T>(), a.grad_v2().end<T>(),
            analytical_a.begin<T>());
        std::copy(b.grad_v2().begin<T>(), b.grad_v2().end<T>(),
            analytical_b.begin<T>());
        auto func1 = [b](mlfe::Tensor & x){
            return matmul(x, b, true);
        };
        auto func2 = [a](mlfe::Tensor & x){
            return matmul(a, x, true);
        };

        auto numerical = numerical_gradient_v2(func1, a, grad_eps);
        auto diff = calculate_gradient_diff<T>(numerical, analytical_a);
        EXPECT_NEAR(diff, T(0), pass_eps);

        numerical = numerical_gradient_v2(func2, b, grad_eps);
        diff = calculate_gradient_diff<T>(numerical, analytical_b);
        EXPECT_NEAR(diff, T(0), pass_eps);
    }

    // // [2, 2] = [3, 2]^T x [2, 3]^T
    {
        auto a = fn::create_variable({3, 2});
        auto b = fn::create_variable({2, 3});
        auto analytical_a = fn::create_variable(a.shape());
        auto analytical_b = fn::create_variable(b.shape());
        std::mt19937 rng;
        std::uniform_real_distribution<T> dist(-1, 1);

        std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
            return dist(rng);
        });
        auto c = matmul(a, b, true, true);
        c.backprop_v2();
        std::copy(a.grad_v2().begin<T>(), a.grad_v2().end<T>(),
            analytical_a.begin<T>());
        std::copy(b.grad_v2().begin<T>(), b.grad_v2().end<T>(),
            analytical_b.begin<T>());
        auto func1 = [b](mlfe::Tensor & x){
            return matmul(x, b, true, true);
        };
        auto func2 = [a](mlfe::Tensor & x){
            return matmul(a, x, true, true);
        };

        auto numerical = numerical_gradient_v2(func1, a, grad_eps);
        auto diff = calculate_gradient_diff<T>(numerical, analytical_a);
        EXPECT_NEAR(diff, T(0), pass_eps);

        numerical = numerical_gradient_v2(func2, b, grad_eps);
        diff = calculate_gradient_diff<T>(numerical, analytical_b);
        EXPECT_NEAR(diff, T(0), pass_eps);
    }
}

