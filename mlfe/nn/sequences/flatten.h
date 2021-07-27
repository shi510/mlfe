#pragma once
#include "mlfe/nn/sequences/sequence.h"

namespace mlfe{
namespace nn{
namespace seq{

template <typename T = void>
struct flatten : sequence
{
    flatten() : sequence("flatten"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        auto size = 1;
        for(auto v : input_shape){
            size *= v;
        }
        return {size};
    }

    Tensor forward(Tensor input, bool train_phase) override{
        auto batch_size = input.shape()[0];
        return input.view({batch_size, input.size() / batch_size});
    }
};


} // namespace sequence
} // namespace nn
} // namespace mlfe
