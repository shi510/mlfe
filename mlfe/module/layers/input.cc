#include "mlfe/module/layers/input.h"
#include "mlfe/operators/matmul.h"

namespace mlfe{
namespace module{
namespace layers{

input::input(std::vector<int> shape, std::string name)
	:layer_impl<input>(name)
{
	_shape = shape;
}

void input::build()
{
	_shape.insert(_shape.begin(), 1);
	_in = add_variable(
		"input",
		_shape,
		false
		);
}

Tensor input::call()
{
	return _in;
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe