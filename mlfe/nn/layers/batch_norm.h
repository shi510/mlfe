#pragma once
#include "mlfe/nn/layers/layer.h"

namespace mlfe{
namespace nn{

class batch_norm2d final : public layer_impl<batch_norm2d>
{
public:
    batch_norm2d();

    batch_norm2d(int32_t num_features, std::string name = "batch_norm2d");

    Tensor call(Tensor input, bool trace_running_status=true);

private:
    Tensor __scales;
    Tensor __biases;
    Tensor __rmean;
    Tensor __rvar;
};

} // end namespace nn
} // end namespace mlfe
