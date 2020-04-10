#include "mlfe/module/layers/layer.h"
#include <sstream>

namespace mlfe{
namespace module{
namespace layers{

layer::layer(std::string name)
	: _layer_name(name)
{
}

std::string layer::get_name()
{
	return _layer_name;
}

Tensor layer::add_variable(
	std::string name,
	std::vector<int> shape,
	bool trainable
){
	using namespace functional;
	auto var_name = make_variable_name(name);
	Tensor var = create_variable(shape, trainable);
	var.set_name(var_name);
	_vars[var_name] = var;
	return var;
}

std::string layer::make_variable_name(std::string name)
{
	std::stringstream ss;
	ss<<_layer_name<<"/"<<name;
	return ss.str();
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe