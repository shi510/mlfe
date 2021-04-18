#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/operators_v2/reduce_mean.h>
#include <mlfe/operators_v2/softmax_cross_entropy.h>
#include <mlfe/operators_v2/matmul.h>
#include <mlfe/operators_v2/conv2d.h>
#include <mlfe/operators_v2/maxpool2d.h>
#include <mlfe/operators_v2/relu.h>

namespace models{
using namespace mlfe;
namespace v2 = mlfe::operators_v2;
namespace fn = mlfe::functional;
/*

struct mnist_conv_net : nn::module{
    nn::conv2d conv1;
    nn::conv2d conv2;
    nn::dropout dropout1;
    nn::dropout dropout2;
    nn::linear fc1;
    nn::linear fc2;

    mnist_conv_net(){
        conv1 = trainable(nn::conv2d(3, 32, {3, 3}, true));
        conv2 = trainable(nn::conv2d(32, 64, {3, 3}, true));
        dropout1 = nn::dropout(0.25);
        dropout2 = nn::dropout(0.5);
        fc1 = trainable(nn::linear());
        fc2 = trainable(nn::linear());
    }

    tensor forward(tensor x){
        x = conv1(x);
        x = relu(x);
        x = dropout1(x);
        x = maxpool2d(x, {2, 2}, {1, 1});
        x = conv2(x);
        x = relu(x);
        x = dropout2(x);
        x = maxpool2d(x, {2, 2}, {1, 1});
        x = fc1(x);
        x = fc2(x);
        return x;
    }
};
*/

} // namespace models