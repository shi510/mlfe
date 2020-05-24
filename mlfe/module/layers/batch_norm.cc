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
	__scales = add_variable("batchnorm_scales", { input_shape[1] }, true);
	__biases = add_variable("batchnorm_biases", { input_shape[1] }, true);
	std::fill(__scales.begin<float>(), __scales.end<float>(), 1.f);
	std::fill(__biases.begin<float>(), __biases.end<float>(), 0.f);
}

Tensor batch_norm::call(Tensor input)
{
	return functional::batch_normalize(input, __scales, __biases);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
