#include <gtest/gtest.h>
#include <mlfe/operators_v2/softmax_cross_entropy.h>
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

namespace fn = mlfe::functional;
using namespace mlfe::operators_v2;

TEST(operator_v2, softmax_xent_logits){
    using T = float;
    constexpr T pass_eps = 1e-3;
    auto logits = fn::create_variable({2, 3});
    logits = std::vector<float>{ 4.f, 2.f, 1.f, 0.f, 5.f, 1.f };
    auto labels = fn::create_variable({2, 3});
    labels = std::vector<float>{ 1.f, 0.f, 0.f, 0.f, 0.8f, 0.2f };
    auto result = softmax_cross_entropy(labels, logits);
    EXPECT_NEAR(result.data<T>()[0], 0.16984f, pass_eps);
    EXPECT_NEAR(result.data<T>()[1], 0.82474f, pass_eps);
}

TEST(operator_v2, softmax_xent_logits_grad_manual){
    using T = float;
    constexpr T pass_eps = 1e-3;
    auto logits = fn::create_variable({2, 3});
    logits = std::vector<float>{ 4.f, 2.f, 1.f, 0.f, 5.f, 1.f };
    auto labels = fn::create_variable({2, 3});
    labels = std::vector<float>{ 1.f, 0.f, 0.f, 0.f, 0.8f, 0.2f };
    auto result = softmax_cross_entropy(labels, logits);
    result = reduce_mean(result);
    result.backprop_v2();
    auto target = (0.16984f + 0.82474f) / 2.f;
    EXPECT_NEAR(result.data<T>()[0], target, pass_eps);
    std::vector<T> target_grad =
        {-0.0781f, 0.0570f, 0.0210f, 0.0032f, 0.0878f, -0.0911f};
    auto grad = logits.grad_v2();
    for(int n = 0; n < grad.size(); ++n)
    {
        EXPECT_NEAR(grad.data<T>()[n], target_grad[n], pass_eps);
    }
}
