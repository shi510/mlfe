#include <gtest/gtest.h>
#include <mlfe/core.h>
#include <mlfe/core/tensor.h>
#include <mlfe/operators_v2/sigmoid.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>
#include <numeric>


using namespace mlfe;
namespace fn = functional;
namespace op = operators_v2;

TEST(operator_v2, sigmoid){
    using T = float;
    auto var = Tensor::from_vector<T>({0, -0.5, 0.5, 1}, {2, 2});
    auto result = op::sigmoid(var);

    EXPECT_EQ(result.data<T>()[0], 0.5);

    EXPECT_LE(result.data<T>()[1], 0.3775 + 1e-4);
    EXPECT_GE(result.data<T>()[1], 0.3775 - 1e-4);

    EXPECT_LE(result.data<T>()[2], 0.6225 + 1e-4);
    EXPECT_GE(result.data<T>()[2], 0.6225 - 1e-4);

    EXPECT_LE(result.data<T>()[3], 0.7311 + 1e-4);
    EXPECT_GE(result.data<T>()[3], 0.7311 - 1e-4);
}

TEST(operator_v2, sigmoid_grad){
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
    auto result = op::sigmoid(x);
    result.backprop_v2();
    std::copy(x.grad_v2().begin<T>(), x.grad_v2().end<T>(), analytical.begin<T>());
    auto func = [](mlfe::Tensor& x){
        return op::sigmoid(x);
    };
    auto numerical = numerical_gradient_v2(func, x, grad_eps);
    auto diff = calculate_gradient_diff<T>(numerical, analytical);
    EXPECT_NEAR(diff, T(0), pass_eps);
}
