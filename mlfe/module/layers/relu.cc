#include "mlfe/module/layers/relu.h"
#include "mlfe/operators/activations.h"
#include <algorithm>

namespace mlfe{
namespace module{
namespace layers{

relu::relu(std::string name)
	: layer_impl<relu>(name)
{
}

void relu::build(std::vector<int> input_shape)
{
}

Tensor relu::call(Tensor input)
{
	return functional::relu(input);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
