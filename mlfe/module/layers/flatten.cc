#include "mlfe/module/layers/flatten.h"
#include "mlfe/operators/basic_arithmetics.h"
#include "mlfe/operators/broadcasting.h"
#include "mlfe/operators/convolution.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace module{
namespace layers{

namespace fn = functional;

flatten::flatten(std::string name)
	: layer_impl<flatten>(name)
{
}

void flatten::build(std::vector<int> input_shape)
{
	__batch = input_shape[0];
	__out_elements = std::accumulate(input_shape.begin() + 1,
		input_shape.end(), 1, std::multiplies<int>());
}

Tensor flatten::call(Tensor input)
{
	return fn::reshape(input, {-1, __out_elements});
}
} // end namespace layer
} // end namespace module
} // end namespace mlfe
