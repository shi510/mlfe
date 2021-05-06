#pragma once
#include <string>
#include <vector>
#include "mlfe/core/tensor.h"

namespace mlfe{
namespace nn{

struct layer
{
    layer() = default;

    layer(std::string name);

    std::string get_name();

    std::vector<Tensor> & traiable_variables();

protected:
    Tensor add_variable(
        std::string name,
        std::vector<int> shape,
        bool trainable = false
    );

    std::string make_variable_name(std::string name);

private:
    std::string _layer_name;
    std::vector<Tensor> __variables;
};

template <typename Derived>
struct layer_impl : layer
{
    layer_impl() = default;

    layer_impl(std::string name);

    template <typename ...Args>
    auto operator()(Args ...args);
};

template <typename Derived>
layer_impl<Derived>::layer_impl(std::string name)
    : layer(name)
{}

template <typename Derived>
template <typename ...Args>
auto layer_impl<Derived>::operator()(Args ...args)
{
    return static_cast<Derived*>(this)->call(args...);
}

} // namespace nn
} // namespace mlfe
