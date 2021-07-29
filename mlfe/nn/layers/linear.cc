#include "mlfe/nn/layers/linear.h"
#include "mlfe/operators/basic_arithmetic.h"
#include "mlfe/operators/matmul.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace nn{

linear::linear(int input_channels,
    int out_channels,
    bool use_bias,
    std::string name)
    : layer_impl<linear>(name)
{
    __use_bias = use_bias;
    auto init_fn = [&]() {
        float std = std::sqrt(6.f / (input_channels));
        auto dist = std::uniform_real_distribution<float>(-std, std);
        return dist(__rng);
    };
    __w = add_variable("weights", { input_channels, out_channels }, true);
    std::generate(__w.begin<float>(), __w.end<float>(), init_fn);
    if(__use_bias)
    {
        __b = add_variable("bias", { out_channels }, true);
        std::fill(__b.begin<float>(), __b.end<float>(), 0.1f);
    }
}

Tensor linear::call(Tensor input)
{
    Tensor y = operators::matmul(input, __w);
    if(__use_bias) { y = y + __b; }
    return y;
}

} // end namespace nn
} // end namespace mlfe
