#pragma once
#include "mlfe/nn/sequences/sequence.h"
#include "mlfe/operators_v2/matmul.h"
#include <cassert>
#include <random>

namespace mlfe{
namespace nn{
namespace seq{

template <int OutChannels, bool use_bias=true>
struct linear : sequence
{
    linear() : sequence("linear"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        assert(input_shape.size() == 1);

        auto init_fn = [&]() {
            float std = std::sqrt(6.f / (input_shape[0]));
            auto dist = std::uniform_real_distribution<float>(-std, std);
            return dist(__rng);
        };
        __weights = add_variable(
            "weights", {input_shape[0], OutChannels}, true);
        std::generate(__weights.begin<float>(), __weights.end<float>(), init_fn);
        if(use_bias)
        {
            __biases = add_variable("bias", { OutChannels }, true);
            std::fill(__biases.begin<float>(), __biases.end<float>(), 0.1f);
        }
        return {OutChannels};
    }

    Tensor forward(Tensor input, bool train_phase) override
    {
        Tensor y = operators_v2::matmul(input, __weights);
        if(use_bias) { y = y + __biases; }
        return y;
    }

    Tensor __weights;
    Tensor __biases;
    std::mt19937 __rng;
};


} // namespace sequence
} // namespace nn
} // namespace mlfe
