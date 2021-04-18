#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/operators_v2/softmax_cross_entropy.h>
#include <mlfe/operators_v2/matmul.h>

namespace models
{
using namespace mlfe;
namespace op = mlfe::operators_v2;
namespace fn = mlfe::functional;

struct mnist_simple_net
{
    mnist_simple_net(int input_size, int out_size)
    {
        this->weights = fn::create_variable({input_size, out_size});
        std::fill(weights.begin<float>(), weights.end<float>(), 0.f);
    }

    Tensor forward(Tensor x)
    {
        auto logits = op::matmul(x, weights);
        return logits;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::softmax_cross_entropy(y_true, y_pred);
        loss = op::reduce_mean(loss);
        return loss;
    }

    void update_weights(float lr)
    {
        weights -= lr * weights.grad_v2();
    }

    void zero_grad()
    {
        weights.grad_v2().zero();
    }

    Tensor weights;
};

} // end namespace models
