#include <gtest/gtest.h>
#include <mlfe/core/tensor.h>
#include <mlfe/operators/softmax_cross_entropy.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/utils/gradient_checker.h>
#include <random>

using namespace mlfe;
namespace fn = mlfe::functional;
using namespace mlfe::operators;

TEST(operator_v2, softmax_xent_logits){
    using T = float;
    constexpr T pass_eps = 1e-3;
    auto logits = Tensor::from_vector<T>({ 4.f, 2.f, 1.f, 0.f, 5.f, 1.f }, {2, 3});
    auto labels = Tensor::from_vector<T>({ 1.f, 0.f, 0.f, 0.f, 0.8f, 0.2f }, {2, 3});
    auto result = softmax_cross_entropy(labels, logits);
    EXPECT_NEAR(result.data<T>()[0], 0.16984f, pass_eps);
    EXPECT_NEAR(result.data<T>()[1], 0.82474f, pass_eps);
}

TEST(operator_v2, softmax_xent_logits_grad_manual){
    using T = float;
    constexpr T pass_eps = 1e-3;
    auto logits = Tensor::from_vector<T>({ 4.f, 2.f, 1.f, 0.f, 5.f, 1.f }, {2, 3});
    auto labels = Tensor::from_vector<T>({ 1.f, 0.f, 0.f, 0.f, 0.8f, 0.2f }, {2, 3});
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
