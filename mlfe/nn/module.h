#pragma once
#include "mlfe/nn/layers/layer.h"
#include <type_traits>
#include <vector>
#include <memory>
#include <algorithm>


namespace mlfe{
namespace nn{

struct module
{
    template <typename T,
        typename = std::enable_if_t<
            std::is_base_of_v<nn::layer, T> || std::is_base_of_v<module, T>
        >
    >
    T trainable(T l){
        auto & list = l.trainable_variables();
        for_each(list.begin(), list.end(),
            [this](Tensor & t){ __trainable_vars.push_back(t); });
        return l;
    }

    std::vector<Tensor> & trainable_variables() {
        return __trainable_vars;
    }

    void zero_grad(){
        std::for_each(
            __trainable_vars.begin(),
            __trainable_vars.end(),
            [](Tensor & v){ v.grad_v2().zero(); });
    }

    template <typename T>
    friend module & operator<<(module & m, const T & seq_layer);

private:
    std::vector<std::unique_ptr<layer>> __seq_vec;
    std::vector<Tensor> __trainable_vars;
};

template <typename T>
module & operator<<(module & m, const T & seq_layer)
{
    m.__seq_vec.push_back(std::make_unique<layer>(seq_layer));
    return m;
}

} // namespace nn
} // namespace mlfe
