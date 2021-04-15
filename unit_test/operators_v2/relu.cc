#include <gtest/gtest.h>
#include <mlfe/operators_v2/relu.h>
#include <mlfe/math/basic_functions.h>
#include <mlfe/utils/gradient_checker.h>
#include <cmath>
#include <random>
#include <chrono>

namespace fn = mlfe::functional;
using namespace mlfe::operators_v2;


TEST(operator_v2, relu){
    using T = float;
    auto input = fn::create_variable({2, 2});
    input = std::vector<float>{-1.35f, 357.5f, -3.5f, 3.15f};
    auto result = relu(input);
    EXPECT_EQ(result.data<T>()[0], 0);
    EXPECT_EQ(result.data<T>()[1], input.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], 0);
    EXPECT_EQ(result.data<T>()[3], input.data<T>()[3]);
}

TEST(operator_v2, relu_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto x = fn::create_variable({2, 2});
    auto analytical = std::vector<T>(x.size());
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(x.begin<T>(), x.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = relu(x);
    result.backprop_v2();
    std::copy(x.grad_v2().begin<T>(), x.grad_v2().end<T>(), analytical.begin());
    auto func = [](mlfe::Tensor& x){
        return relu(x);
    };
    auto numerical = numerical_gradient_v2(func, x, grad_eps);
    auto diff = calculate_gradient_diff(numerical, analytical);
    EXPECT_NEAR(diff, T(0), pass_eps);
    for(int n = 0; n < x.size(); ++n){
        auto diff = std::abs(analytical.data()[n] - numerical.data<T>()[n]);
        EXPECT_LE(diff, pass_eps);
        EXPECT_GE(diff, -pass_eps);
    }
}
