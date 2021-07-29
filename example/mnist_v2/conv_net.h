#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/softmax_cross_entropy.h>
#include <mlfe/operators/maxpool2d.h>
#include <mlfe/operators/relu.h>
#include <mlfe/nn/layers/linear.h>
#include <mlfe/nn/layers/conv2d.h>
#include <mlfe/operators/dropout.h>
#include <mlfe/nn/module.h>

namespace models{
using namespace mlfe;
namespace op = mlfe::operators;

struct mnist_conv_net : nn::module{
    nn::conv2d conv1;
    nn::conv2d conv2;
    nn::linear fc1;
    nn::linear fc2;

    mnist_conv_net(){
        conv1 = trainable(nn::conv2d(1, 16, {3, 3}, {1, 1}, true));
        conv2 = trainable(nn::conv2d(16, 32, {3, 3}, {1, 1}, true));
        fc1 = trainable(nn::linear(7*7*32, 256));
        fc2 = trainable(nn::linear(256, 10));
    }

    Tensor forward(Tensor x, bool is_training=false){
        x = conv1(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::relu(x);
        x = conv2(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::relu(x);
        x = op::dropout(x, 0.5, is_training);
        x = x.view({x.shape()[0], 7 * 7 * 32});
        x = fc1(x);
        x = op::relu(x);
        x = fc2(x);
        return x;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::softmax_cross_entropy(y_true, y_pred);
        loss = op::reduce_mean(loss);
        return loss;
    }
};

} // namespace models
