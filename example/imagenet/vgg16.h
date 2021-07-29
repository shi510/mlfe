#pragma once
#include <mlfe/core/tensor.h>
#include <mlfe/nn/module.h>
#include <mlfe/nn/sequences/batch_norm.h>
#include <mlfe/nn/sequences/conv2d.h>
#include <mlfe/nn/sequences/flatten.h>
#include <mlfe/nn/sequences/maxpool2d.h>
#include <mlfe/nn/sequences/linear.h>
#include <mlfe/nn/sequences/relu.h>

namespace models{
using namespace mlfe;
namespace seq = mlfe::nn::seq;

struct vgg16 : nn::module{
    nn::module net_block;

    template <int C>
    nn::module conv_block(){
        nn::module m;
        return m
            << seq::conv2d<C, size<3, 3>, size<1, 1>, true>()
            << seq::batch_norm2d<>() << seq::relu<>();
    }

    vgg16(){
        net_block
            << conv_block<64>() << conv_block<64>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<128>() << conv_block<128>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<256>() << conv_block<256>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << conv_block<512>() << conv_block<512>()
            << seq::maxpool2d<size<2, 2>, size<2, 2>>()

            << seq::flatten<>()

            << seq::linear<4096>() << seq::batch_norm1d<>() << seq::relu<>()
            << seq::linear<4096>() << seq::batch_norm1d<>() << seq::relu<>()
            << seq::linear<1000>();
        net_block.build({224, 224, 3});
        net_block = trainable(net_block);
    }

    Tensor forward(Tensor x, bool is_train){
        x = net_block(x, is_train);
        return x;
    }
};

} // namespace models
