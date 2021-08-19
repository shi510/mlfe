#include <gtest/gtest.h>
#include <mlfe/operators/basic_arithmetic.h>
#include <mlfe/operators/broadcast.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>
#include <iostream>
#include <sstream>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator, elementwise_add){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_vector<T>({5, 6, 7, 8}, {2, 2});
    auto result = add(a, b);
    EXPECT_EQ(result.data<T>()[0], 6);
    EXPECT_EQ(result.data<T>()[1], 8);
    EXPECT_EQ(result.data<T>()[2], 10);
    EXPECT_EQ(result.data<T>()[3], 12);
}

TEST(operator, elementwise_add_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = fn::create_variable(shape);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = add(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return add(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return add(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}


TEST(operator, elementwise_sub){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_vector<T>({5, 6, 7, 8}, {2, 2});
    auto result = sub(a, b);
    EXPECT_EQ(result.data<T>()[0], -4);
    EXPECT_EQ(result.data<T>()[1], -4);
    EXPECT_EQ(result.data<T>()[2], -4);
    EXPECT_EQ(result.data<T>()[3], -4);
}

TEST(operator, elementwise_sub_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = fn::create_variable(shape);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = sub(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return sub(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return sub(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, elementwise_mul){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_vector<T>({5, 6, 7, 8}, {2, 2});
    auto result = mul(a, b);
    EXPECT_EQ(result.data<T>()[0], 5);
    EXPECT_EQ(result.data<T>()[1], 12);
    EXPECT_EQ(result.data<T>()[2], 21);
    EXPECT_EQ(result.data<T>()[3], 32);
}

TEST(operator, elementwise_mul_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = fn::create_variable(shape);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = mul(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return mul(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return mul(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, elementwise_div){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_vector<T>({5, 6, 7, 8}, {2, 2});
    auto result = div(a, b);
    EXPECT_EQ(result.data<T>()[0], 1.f/5.f);
    EXPECT_EQ(result.data<T>()[1], 2.f/6.f);
    EXPECT_EQ(result.data<T>()[2], 3.f/7.f);
    EXPECT_EQ(result.data<T>()[3], 4.f/8.f);
}

TEST(operator, elementwise_div_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = fn::create_variable(shape);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable(shape);
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = div(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return div(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return div(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, scalar_vector_add){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_scalar<T>(5);
    auto result = add(a, b);
    EXPECT_EQ(result.data<T>()[0], 6);
    EXPECT_EQ(result.data<T>()[1], 7);
    EXPECT_EQ(result.data<T>()[2], 8);
    EXPECT_EQ(result.data<T>()[3], 9);
}

TEST(operator, scalar_vector_add_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = Tensor::from_scalar<T>(0);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable({});
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = add(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return add(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return add(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, scalar_vector_sub){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_scalar<T>(5);
    auto result = sub(a, b);
    EXPECT_EQ(result.data<T>()[0], -4);
    EXPECT_EQ(result.data<T>()[1], -3);
    EXPECT_EQ(result.data<T>()[2], -2);
    EXPECT_EQ(result.data<T>()[3], -1);
}

TEST(operator, scalar_vector_sub_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = Tensor::from_scalar<T>(0);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable({});
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = sub(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return sub(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return sub(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, scalar_vector_mul){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_scalar<T>(5);
    auto result = mul(a, b);
    EXPECT_EQ(result.data<T>()[0], 5);
    EXPECT_EQ(result.data<T>()[1], 10);
    EXPECT_EQ(result.data<T>()[2], 15);
    EXPECT_EQ(result.data<T>()[3], 20);
}

TEST(operator, scalar_vector_mul_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = Tensor::from_scalar<T>(0);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable({});
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    std::generate(b.begin<T>(), b.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = mul(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return mul(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return mul(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, scalar_vector_div){
    using T = float;
    auto a = Tensor::from_vector<T>({10, 15, 20, 25}, {2, 2});
    auto b = Tensor::from_scalar<T>(5);
    auto result = div(a, b);
    EXPECT_EQ(result.data<T>()[0], 2);
    EXPECT_EQ(result.data<T>()[1], 3);
    EXPECT_EQ(result.data<T>()[2], 4);
    EXPECT_EQ(result.data<T>()[3], 5);
}

TEST(operator, scalar_vector_div_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto shape = std::vector<int>{2, 2};
    auto a = fn::create_variable(shape);
    auto b = Tensor::from_scalar<T>(2);
    auto analytical_a = fn::create_variable(shape);
    auto analytical_b = fn::create_variable({});
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(a.begin<T>(), a.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = div(a, b);
    result.backprop();
    std::copy(a.grad().begin<T>(), a.grad().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad().begin<T>(), b.grad().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](mlfe::Tensor& a){
        return div(a, b);
    };
    auto func_b = [a](mlfe::Tensor& b){
        return div(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}

TEST(operator, broadcast_add){
    using T = float;
    //
    // { {1, 2},                   { {6, 12},
    //   {3, 4},    +  {5, 10}  =    {8, 14},
    //   {5, 6} }                    {10, 16} }
    //
    auto a = Tensor::from_vector<T>({1, 2, 3, 4, 5, 6}, {3, 2});
    auto b = Tensor::from_vector<T>({5, 10}, {2});
    auto result = add(a, b);
    EXPECT_EQ(result.shape().size(), 2);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 2);
    EXPECT_EQ(result.data<T>()[0], 6);
    EXPECT_EQ(result.data<T>()[1], 12);
    EXPECT_EQ(result.data<T>()[2], 8);
    EXPECT_EQ(result.data<T>()[3], 14);
    EXPECT_EQ(result.data<T>()[4], 10);
    EXPECT_EQ(result.data<T>()[5], 16);
}