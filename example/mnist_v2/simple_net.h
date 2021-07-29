#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/softmax_cross_entropy.h>
#include <mlfe/operators/matmul.h>

namespace models
{
using namespace mlfe;
namespace op = mlfe::operators;
namespace fn = mlfe::functional;

struct mnist_simple_net
{
    Tensor weights;
    Tensor biases;

    mnist_simple_net(int input_size, int out_size)
    {
        weights = fn::create_variable({input_size, out_size});
        biases = fn::create_variable({out_size});
        std::fill(weights.begin<float>(), weights.end<float>(), 0.f);
        std::fill(biases.begin<float>(), biases.end<float>(), 0.1f);
    }

    Tensor forward(Tensor x)
    {
        auto logits = op::matmul(x, weights);
        return logits + biases;
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
        biases -= 2.f * lr * biases.grad_v2();
    }

    void zero_grad()
    {
        weights.grad_v2().zero();
        biases.grad_v2().zero();
    }
};

} // end namespace models
