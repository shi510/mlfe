#include "mlfe/module/layers/sigmoid.h"
#include "mlfe/operators/activations.h"

namespace mlfe{
namespace module{
namespace layers{

sigmoid::sigmoid(std::string name)
	: layer_impl<sigmoid>(name)
{
}

void sigmoid::build(std::vector<int> input_shape)
{
}

Tensor sigmoid::call(Tensor input)
{
	return functional::sigmoid(input);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
