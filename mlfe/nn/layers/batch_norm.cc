#include "mlfe/nn/layers/batch_norm.h"
#include "mlfe/operators/basic_arithmetic.h"
#include "mlfe/operators/batch_norm2d.h"
#include "mlfe/operators/batch_norm1d.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace nn{

batch_norm2d::batch_norm2d(){}

batch_norm2d::batch_norm2d(
    int32_t num_features,
    std::string name) : layer_impl<batch_norm2d>(name)
{
    __scales = add_variable("scales", { num_features }, true);
    __biases = add_variable("biases", { num_features }, true);
    __rmean = add_variable("running_mean", { num_features }, false);
    __rvar = add_variable("running_var", { num_features }, false);
    std::fill(__scales.begin<float>(), __scales.end<float>(), 1.f);
    std::fill(__biases.begin<float>(), __biases.end<float>(), 0.f);
    std::fill(__rmean.begin<float>(), __rmean.end<float>(), 0.f);
    std::fill(__rvar.begin<float>(), __rvar.end<float>(), 0.f);
}

Tensor batch_norm2d::call(Tensor input, bool trace_running_status)
{
    Tensor y = operators::batch_norm2d(
        input, __scales, __biases, __rmean, __rvar, trace_running_status);
    return y;
}

batch_norm1d::batch_norm1d(){}

batch_norm1d::batch_norm1d(
    int32_t num_features,
    std::string name) : layer_impl<batch_norm1d>(name)
{
    __scales = add_variable("scales", { num_features }, true);
    __biases = add_variable("biases", { num_features }, true);
    __rmean = add_variable("running_mean", { num_features }, false);
    __rvar = add_variable("running_var", { num_features }, false);
    std::fill(__scales.begin<float>(), __scales.end<float>(), 1.f);
    std::fill(__biases.begin<float>(), __biases.end<float>(), 0.f);
    std::fill(__rmean.begin<float>(), __rmean.end<float>(), 0.f);
    std::fill(__rvar.begin<float>(), __rvar.end<float>(), 0.f);
}

Tensor batch_norm1d::call(Tensor input, bool trace_running_status)
{
    Tensor y = operators::batch_norm1d(
        input, __scales, __biases, __rmean, __rvar, trace_running_status);
    return y;
}

} // end namespace nn
} // end namespace mlfe
