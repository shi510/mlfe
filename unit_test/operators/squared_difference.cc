#include <gtest/gtest.h>
#include <mlfe/operators/squared_difference.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator, squared_difference_fwd){
    using T = float;
    auto a = Tensor::from_vector<T>({1, 2, 3, -4}, {2, 2});
    auto b = Tensor::from_vector<T>({5, 4, 3, 2}, {2, 2});
    auto y = squared_difference(a, b);
    EXPECT_EQ(y.data<T>()[0], 16);
    EXPECT_EQ(y.data<T>()[1], 4);
    EXPECT_EQ(y.data<T>()[2], 0);
    EXPECT_EQ(y.data<T>()[3], 36);
}

TEST(operator, squared_difference_bwd){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto a = fn::create_variable({2, 2});
    auto b = fn::create_variable({2, 2});
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
    auto y = squared_difference(a, b);
    y.backprop_v2();
    std::copy(a.grad_v2().begin<T>(), a.grad_v2().end<T>(), analytical_a.begin<T>());
    std::copy(b.grad_v2().begin<T>(), b.grad_v2().end<T>(), analytical_b.begin<T>());
    auto func_a = [b](Tensor& a){
        return squared_difference(a, b);
    };
    auto func_b = [a](Tensor& b){
        return squared_difference(a, b);
    };
    auto numerical_a = numerical_gradient_v2(func_a, a, grad_eps);
    auto numerical_b = numerical_gradient_v2(func_b, b, grad_eps);
    auto diff_a = calculate_gradient_diff<T>(numerical_a, analytical_a);
    auto diff_b = calculate_gradient_diff<T>(numerical_b, analytical_b);
    EXPECT_NEAR(diff_a, T(0), pass_eps);
    EXPECT_NEAR(diff_b, T(0), pass_eps);
}
