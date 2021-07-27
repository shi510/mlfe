#pragma once
#include <string>
#include "mlfe/nn/layers/layer.h"

namespace mlfe{

template<int... Is>
struct size {
    static std::vector<int> to_vector()
    {
        return {Is...};
    }
};

namespace nn{
namespace seq{

struct sequence : layer
{
    sequence(std::string name) : layer(name) { }

    virtual Tensor forward(Tensor input, bool train_phase=false) = 0;

    virtual std::vector<int> build(std::vector<int> shape) = 0;

    Tensor operator()(Tensor input, bool train_phase){
        return this->forward(input, train_phase);
    }
};


} // namespace seq
} // namespace nn
} // namespace mlfe
