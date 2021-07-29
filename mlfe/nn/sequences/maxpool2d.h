
#pragma once
#include "mlfe/nn/sequences/sequence.h"
#include "mlfe/operators/maxpool2d.h"
#include "mlfe/operators/utils.h"
#include <cassert>

namespace mlfe{
namespace nn{
namespace seq{

template <typename K, typename S>
struct maxpool2d : sequence
{
    maxpool2d() : sequence("maxpool2d"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        assert(input_shape.size() == 3);
        __kernel = K().to_vector();
        __stride = S().to_vector();
        auto oh = operators::utils::calc_conv_output(
            input_shape[0], __kernel[0], __stride[0], 0);
        auto ow = operators::utils::calc_conv_output(
            input_shape[1], __kernel[1], __stride[1], 0);
        return {oh, ow, input_shape[2]};
    }

    Tensor forward(Tensor input, bool train_phase) override
    {
        return operators::maxpool2d(input, __kernel, __stride);
    }

    std::vector<int> __kernel;
    std::vector<int> __stride;
};

} // namespace sequence
} // namespace nn
} // namespace mlfe
