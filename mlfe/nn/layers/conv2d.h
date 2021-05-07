#pragma once
#include "mlfe/nn/layers/layer.h"
#include <random>
#include <vector>

namespace mlfe{
namespace nn{

class conv2d final : public layer_impl<conv2d>
{
public:
    conv2d() = default;

    conv2d(int input_channels,
        int out_channels,
        std::vector<int> kernel,
        std::vector<int> stride,
        bool same_out=false,
        bool use_bias = true,
        std::string name = "conv2d");

    Tensor call(Tensor input);

private:
    std::vector<int> __stride;
    bool __same_out;
    bool __use_bias;
    Tensor __w, __b;
    std::mt19937 __rng;
};

} // end namespace nn
} // end namespace mlfe
