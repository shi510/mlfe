#include <gtest/gtest.h>
#include <mlfe/core/tensor.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
using namespace mlfe::operators;
namespace fn = mlfe::functional;

TEST(operator_v2, reduce_mean){
    using T = float;
    constexpr T pass_eps = 1e-3;
    auto input = Tensor::from_vector<T>({-1.35f, 357.5f, -3.5f, 3.15f}, {2, 2});
    auto result = reduce_mean(input);
    EXPECT_EQ(result.size(), 1);
    EXPECT_NEAR(result.data<T>()[0], 88.95f, pass_eps);
}

TEST(operator_v2, reduce_mean_grad){
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
    auto result = reduce_mean(x);
    result.backprop_v2();
    std::copy(x.grad_v2().begin<T>(), x.grad_v2().end<T>(), analytical.begin<T>());
    auto func = [](mlfe::Tensor& x){
        return reduce_mean(x);
    };
    auto numerical = numerical_gradient_v2(func, x, grad_eps);
    auto diff = calculate_gradient_diff<T>(numerical, analytical);
    EXPECT_NEAR(diff, T(0), pass_eps);
    for(int n = 0; n < x.size(); ++n){
        auto diff = std::abs(analytical.data<T>()[n] - numerical.data<T>()[n]);
        EXPECT_LE(diff, pass_eps);
        EXPECT_GE(diff, -pass_eps);
    }
}
