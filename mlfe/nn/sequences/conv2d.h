
#pragma once
#include "mlfe/nn/sequences/sequence.h"
#include "mlfe/operators_v2/conv2d.h"
#include "mlfe/operators_v2/utils.h"
#include <cassert>
#include <random>

namespace mlfe{
namespace nn{
namespace seq{

template <int out_channels, typename K, typename S, bool P, bool use_bias=true>
struct conv2d : sequence
{
    conv2d() : sequence("conv2d"){
        __same_out = P;
        __use_bias = use_bias;
    }

    std::vector<int> build(std::vector<int> input_shape) override
    {
        assert(input_shape.size() == 3);
        int input_channels = input_shape[2];
        auto kernel = K().to_vector();
        __stride = S().to_vector();
        auto init_fn = [&]() {
            float std = std::sqrt(6.f / (kernel[0] * kernel[1] * input_channels));
            auto dist = std::uniform_real_distribution<float>(-std, std);
            return dist(__rng);
        };
        __w = this->add_variable("weights",
            { kernel[0], kernel[1], input_channels, out_channels }, true);
        std::generate(__w.begin<float>(), __w.end<float>(), init_fn);
        if(__use_bias)
        {
            __b = this->add_variable("bias", { out_channels }, true);
            std::fill(__b.begin<float>(), __b.end<float>(), 0.1f);
        }
        int padh = 0;
        int padw = 0;
        if(P){
            padh = operators_v2::utils::calc_conv_same_output_padding_size(
                input_shape[0], kernel[0], __stride[0]);
            padw = operators_v2::utils::calc_conv_same_output_padding_size(
                input_shape[1], kernel[1], __stride[1]);
        }
        auto oh = operators_v2::utils::calc_conv_output(
            input_shape[0], kernel[0], __stride[0], padh);
        auto ow = operators_v2::utils::calc_conv_output(
            input_shape[1], kernel[1], __stride[1], padw);
        return {oh, ow, out_channels};
    }

    Tensor forward(Tensor input, bool train_phase) override{
        Tensor y = operators_v2::conv2d(input, __w, __stride, __same_out);
        if(__use_bias) { y = y + __b; }
        return y;
    }

    std::vector<int> __stride;
    bool __use_bias;
    bool __same_out;
    Tensor __w, __b;
    std::mt19937 __rng;
};

} // namespace sequence
} // namespace nn
} // namespace mlfe
