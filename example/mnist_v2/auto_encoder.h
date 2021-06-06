#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/operators_v2/squared_difference.h>
#include <mlfe/operators_v2/sigmoid.h>
#include <mlfe/nn/layers/linear.h>
#include <mlfe/nn/module.h>

namespace models{
using namespace mlfe;
namespace op = mlfe::operators_v2;

struct encoder : nn::module{
    nn::linear fc1;
    nn::linear fc2;
    nn::linear fc3;
    nn::linear fc4;

    encoder(){
        fc1 = trainable(nn::linear(28*28, 300));
        fc2 = trainable(nn::linear(300, 150));
        fc3 = trainable(nn::linear(150, 50));
        fc4 = trainable(nn::linear(50, 10));
    }

    Tensor forward(Tensor x){
        x = fc1(x);
        x = op::sigmoid(x);
        x = fc2(x);
        x = op::sigmoid(x);
        x = fc3(x);
        x = op::sigmoid(x);
        x = fc4(x);
        x = op::sigmoid(x);
        return x;
    }
};

struct decoder : nn::module{
    nn::linear fc1;
    nn::linear fc2;
    nn::linear fc3;
    nn::linear fc4;

    decoder(){
        fc1 = trainable(nn::linear(10, 50));
        fc2 = trainable(nn::linear(50, 150));
        fc3 = trainable(nn::linear(150, 300));
        fc4 = trainable(nn::linear(300, 28*28));
    }

    Tensor forward(Tensor x){
        x = fc1(x);
        x = op::sigmoid(x);
        x = fc2(x);
        x = op::sigmoid(x);
        x = fc3(x);
        x = op::sigmoid(x);
        x = fc4(x);
        x = op::sigmoid(x);
        return x;
    }
};

struct autoencoder : nn::module{
    encoder encode;
    decoder decode;

    autoencoder()
    {
        encode = trainable(encoder());
        decode = trainable(decoder());
    }

    Tensor forward(Tensor x)
    {
        x = encode.forward(x);
        x = decode.forward(x);
        return x;
    }

    Tensor criterion(Tensor y_true, Tensor y_pred)
    {
        auto loss = op::squared_difference(y_true, y_pred);
        return op::reduce_mean(loss);
    }
};

} // namespace models
