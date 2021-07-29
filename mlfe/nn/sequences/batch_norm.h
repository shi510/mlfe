#pragma once
#include "mlfe/nn/sequences/sequence.h"
#include "mlfe/operators_v2/batch_norm1d.h"
#include "mlfe/operators_v2/batch_norm2d.h"
#include <cassert>

namespace mlfe{
namespace nn{
namespace seq{

template <typename T = void>
struct batch_norm2d : sequence
{
    batch_norm2d() : sequence("batch_norm2d"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        assert(input_shape.size() == 3);
        __scales = add_variable("scales", { input_shape[2] }, true);
        __biases = add_variable("biases", { input_shape[2] }, true);
        __rmean = add_variable("running_mean", { input_shape[2] }, false);
        __rvar = add_variable("running_var", { input_shape[2] }, false);
        std::fill(__scales.begin<float>(), __scales.end<float>(), 1.f);
        std::fill(__biases.begin<float>(), __biases.end<float>(), 0.f);
        std::fill(__rmean.begin<float>(), __rmean.end<float>(), 0.f);
        std::fill(__rvar.begin<float>(), __rvar.end<float>(), 0.f);
        return input_shape;
    }

    Tensor forward(Tensor input, bool train_phase) override{
       return operators_v2::batch_norm2d(
        input, __scales, __biases, __rmean, __rvar, train_phase);
    }

    Tensor __scales;
    Tensor __biases;
    Tensor __rmean;
    Tensor __rvar;
};

template <typename T = void>
struct batch_norm1d : sequence
{
    batch_norm1d() : sequence("batch_norm1d"){}

    std::vector<int> build(std::vector<int> input_shape) override
    {
        assert(input_shape.size() == 1);
        __scales = add_variable("scales", { input_shape[0] }, true);
        __biases = add_variable("biases", { input_shape[0] }, true);
        __rmean = add_variable("running_mean", { input_shape[0] }, false);
        __rvar = add_variable("running_var", { input_shape[0] }, false);
        std::fill(__scales.begin<float>(), __scales.end<float>(), 1.f);
        std::fill(__biases.begin<float>(), __biases.end<float>(), 0.f);
        std::fill(__rmean.begin<float>(), __rmean.end<float>(), 0.f);
        std::fill(__rvar.begin<float>(), __rvar.end<float>(), 0.f);
        return input_shape;
    }

    Tensor forward(Tensor input, bool train_phase) override{
       return operators_v2::batch_norm1d(
        input, __scales, __biases, __rmean, __rvar, train_phase);
    }

    Tensor __scales;
    Tensor __biases;
    Tensor __rmean;
    Tensor __rvar;
};


} // namespace sequence
} // namespace nn
} // namespace mlfe
