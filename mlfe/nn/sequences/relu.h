#pragma once
#include "mlfe/nn/sequences/sequence.h"
#include "mlfe/operators/relu.h"

namespace mlfe{
namespace nn{
namespace seq{

template <typename T = void>
struct relu : sequence
{
    relu() : sequence("relu"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        return input_shape;
    }

    Tensor forward(Tensor input, bool train_phase) override
    {
       return operators::relu(input);
    }
};


} // namespace sequence
} // namespace nn
} // namespace mlfe
