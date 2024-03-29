#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators/basic_arithmetic.h>
#include <mlfe/operators/reduce_mean.h>
#include <mlfe/operators/softmax_cross_entropy.h>
#include <mlfe/operators/maxpool2d.h>
#include <mlfe/operators/relu.h>
#include <mlfe/operators/global_avg_pool2d.h>
#include <mlfe/nn/layers/linear.h>
#include <mlfe/nn/layers/conv2d.h>
#include <mlfe/nn/layers/batch_norm.h>
#include <mlfe/operators/dropout.h>
#include <mlfe/nn/module.h>
#include <iostream>

namespace models{
using namespace mlfe;
namespace op = mlfe::operators;

struct conv_block : nn::module{
    nn::conv2d conv1;
    nn::batch_norm2d bn1;
    nn::conv2d conv2;
    nn::batch_norm2d bn2;

    conv_block(){}

    conv_block(int32_t in_chann, int32_t out_chann){
        conv1 = trainable(nn::conv2d(in_chann, out_chann,
            /*kernel_size=*/{3, 3}, /*strides=*/{1, 1},
            /*same_out=*/true, /*use_bias=*/false));
        bn1 = trainable(nn::batch_norm2d(out_chann));
        conv2 = trainable(nn::conv2d(out_chann, out_chann,{3, 3}, {1, 1}, true, false));
        bn2 = trainable(nn::batch_norm2d(out_chann));
    }

    Tensor operator()(Tensor x, float drop_rate, bool is_training=false){
        x = conv1(x);
        x = bn1(x, is_training);
        x = op::relu(x);
        x = conv2(x);
        x = bn2(x, is_training);
        x = op::relu(x);
        x = op::maxpool2d(x, {2, 2}, {2, 2});
        x = op::dropout(x, drop_rate, is_training);
        return x;
    }
};

struct cifar10_convnet : nn::module{
    conv_block block1;
    conv_block block2;
    conv_block block3;
    nn::linear fc1;
    nn::batch_norm1d bn1;
    nn::linear fc2;

    cifar10_convnet(){
        block1 = trainable(conv_block(3, 64));
        block2 = trainable(conv_block(64, 128));
        block3 = trainable(conv_block(128, 256));
        fc1 = trainable(nn::linear(4*4*256, 512, /*use_bias=*/false));
        bn1 = trainable(nn::batch_norm1d(512));
        fc2 = trainable(nn::linear(512, 10));
    }

    Tensor forward(Tensor x, bool is_training=false){
        x = block1(x, 0.3, is_training);
        x = block2(x, 0.4, is_training);
        x = block3(x, 0.5, is_training);
        x = x.view({x.shape()[0], 4*4*256});
        x = fc1(x);
        x = bn1(x, is_training);
        x = op::relu(x);
        x = op::dropout(x, 0.5, is_training);
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
