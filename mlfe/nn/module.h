#pragma once
#include "mlfe/nn/layers/layer.h"
#include "mlfe/nn/sequences/sequence.h"
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
        auto & list = l.variables();
        for_each(list.begin(), list.end(), [this](Tensor & t){
            if(t.trainable()) __trainable_vars.push_back(t);
            __vars.push_back(t);});
        return l;
    }

    std::vector<Tensor> & trainable_variables() {
        return __trainable_vars;
    }
    
    std::vector<Tensor> & variables() {
        return __vars;
    }

    void zero_grad(){
        std::for_each(
            __trainable_vars.begin(),
            __trainable_vars.end(),
            [](Tensor & v){ v.grad_v2().zero(); });
    }

    virtual Tensor operator()(Tensor x, bool train_phase=false){
        for(auto& l : __seq_vec){
            x = l->forward(x, train_phase);
        }
        return x;
    }

    void build(std::vector<int> shape)
    {
        for(auto &l : __seq_vec)
        {
            shape = l->build(shape);
            for(auto v : l->variables()){
                __vars.push_back(v);
            }
        }
    }

    template <typename T>
    friend module & operator<<(module & m, const T seq_layer);

private:
    std::vector<std::shared_ptr<seq::sequence>> __seq_vec;
    std::vector<Tensor> __trainable_vars;
    std::vector<Tensor> __vars;
};

template <typename T>
module & operator<<(module & m, const T layer_or_module)
{
    if constexpr (std::is_base_of<seq::sequence, T>::value)
    {
        m.__seq_vec.push_back(std::make_shared<T>(layer_or_module));
    }
    else if constexpr (std::is_base_of<module, T>::value)
    {
        for(auto &l : layer_or_module.__seq_vec)
        {
            m.__seq_vec.push_back(l);
        }
    }
    return m;
}

} // namespace nn
} // namespace mlfe
