#include <gtest/gtest.h>
#include <mlfe/operators/relu.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator, relu){
    using T = float;
    auto input = Tensor::from_vector<T>({-1.35f, 357.5f, -3.5f, 3.15f}, {2, 2});
    auto result = relu(input);
    EXPECT_EQ(result.data<T>()[0], 0);
    EXPECT_EQ(result.data<T>()[1], input.data<T>()[1]);
    EXPECT_EQ(result.data<T>()[2], 0);
    EXPECT_EQ(result.data<T>()[3], input.data<T>()[3]);
}

TEST(operator, relu_grad){
    using T = float;
    constexpr T grad_eps = 1e-4;
    constexpr T pass_eps = 1e-3;
    auto x = fn::create_variable({2, 2});
    auto analytical = fn::create_variable(x.shape());
    std::mt19937 rng;
    std::uniform_real_distribution<T> dist(-1, 1);

    std::generate(x.begin<T>(), x.end<T>(), [&rng, &dist](){
        return dist(rng);
    });
    auto result = relu(x);
    result.backprop();
    std::copy(x.grad().begin<T>(), x.grad().end<T>(), analytical.begin<T>());
    auto func = [](mlfe::Tensor& x){
        return relu(x);
    };
    auto numerical = numerical_gradient_v2(func, x, grad_eps);
    auto diff = calculate_gradient_diff<T>(numerical, analytical);
    EXPECT_NEAR(diff, T(0), pass_eps);
}
