#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/operators_v2/softmax_cross_entropy.h>
#include <mlfe/operators_v2/maxpool2d.h>
#include <mlfe/operators_v2/relu.h>
#include <mlfe/nn/layers/linear.h>
#include <mlfe/nn/layers/conv2d.h>
#include <mlfe/nn/module.h>

namespace models{
using namespace mlfe;
namespace op = mlfe::operators_v2;

struct mnist_conv_net : nn::module{
    nn::conv2d conv1;
    nn::conv2d conv2;
    nn::conv2d conv3;
    nn::linear fc1;
    nn::linear fc2;

    mnist_conv_net(){
        conv1 = trainable(nn::conv2d(1, 12, {3, 3}, {1, 1}, true));
        conv2 = trainable(nn::conv2d(12, 24, {3, 3}, {1, 1}, true));
        fc1 = trainable(nn::linear(7*7*24, 64));
        fc2 = trainable(nn::linear(64, 10));
    }

    Tensor forward(Tensor x){
        x = conv1(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::relu(x);
        x = conv2(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::relu(x);
        x = x.view({x.shape()[0], 7 * 7 * 24});
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
