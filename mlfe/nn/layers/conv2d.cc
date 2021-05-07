#include "mlfe/nn/layers/conv2d.h"
#include "mlfe/operators_v2/basic_arithmetic.h"
#include "mlfe/operators_v2/broadcast.h"
#include "mlfe/operators_v2/conv2d.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace nn{

namespace fn = functional;

conv2d::conv2d(int input_channels,
    int out_channels,
    std::vector<int> kernel,
    std::vector<int> stride,
    bool same_out,
    bool use_bias,
    std::string name)
    : layer_impl<conv2d>(name)
{
    __stride = stride;
    __same_out = same_out;
    __use_bias = use_bias;
    auto init_fn = [&]() {
        float std = std::sqrt(6.f / (kernel[0] * kernel[1] * input_channels));
        auto dist = std::uniform_real_distribution<float>(-std, std);
        return dist(__rng);
    };
    __w = add_variable("weights",
        { kernel[0], kernel[1], input_channels, out_channels }, true);
    std::generate(__w.begin<float>(), __w.end<float>(), init_fn);
    if(__use_bias)
    {
        __b = add_variable("bias", { out_channels }, true);
        std::fill(__b.begin<float>(), __b.end<float>(), 0.1f);
    }
}

Tensor conv2d::call(Tensor input)
{
    Tensor y = operators_v2::conv2d(input, __w, __stride, __same_out);
    if(__use_bias) { y = y + __b; }
    return y;
}

} // end namespace nn
} // end namespace mlfe
