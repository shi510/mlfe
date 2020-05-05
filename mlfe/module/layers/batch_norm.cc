#include "mlfe/module/layers/batch_norm.h"
#include "mlfe/operators/batch_norm.h"
#include <algorithm>

namespace mlfe {
namespace module {
namespace layers {

batch_norm::batch_norm(std::string name)
	: layer_impl<batch_norm>(name)
{
}

void batch_norm::build(std::vector<int> input_shape)
{
}

Tensor batch_norm::call(Tensor input)
{
	return functional::batch_normalize(input);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
