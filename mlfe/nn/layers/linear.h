#pragma once
#include "mlfe/nn/layers/layer.h"
#include <random>

namespace mlfe{
namespace nn{

class linear final : public layer_impl<linear>
{
public:
    linear() = default;

    linear(int input_channels,
        int out_channels,
        bool use_bias = true,
        std::string name = "linear");

    Tensor call(Tensor input);

private:
    bool __use_bias;
    Tensor __w, __b;
    std::mt19937 __rng;
};

} // end namespace nn
} // end namespace mlfe
