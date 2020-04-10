#pragma once
#include "mlfe/core/tensor.h"
#include <map>
#include <string>

namespace mlfe{
namespace module{
namespace layers{

class layer
{
public:
	layer(std::string name);

	std::string get_name();

protected:
	Tensor add_variable(
		std::string name,
		std::vector<int> shape,
		bool trainable = false
	);

	std::string make_variable_name(std::string name);

private:
	std::string _layer_name;
	std::map<std::string, Tensor> _vars;
}; // end class layer

template <typename Derived>
class layer_impl : public layer
{
public:
	layer_impl(std::string name);

	template <typename ...Args>
	void build(Args ...args);

	template <typename ...Args>
	auto operator()(Args ...args);
}; // end class layer_impl<>

template <typename Derived>
layer_impl<Derived>::layer_impl(std::string name)
	: layer(name)
{}

template <typename Derived>
template <typename ...Args>
void layer_impl<Derived>::build(Args ...args)
{
	static_cast<Derived*>(this)->build(args...);
}

template <typename Derived>
template <typename ...Args>
auto layer_impl<Derived>::operator()(Args ...args)
{
	build(args.shape()...);
	return static_cast<Derived*>(this)->call(args...);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe