#include "mlfe/module/layers/dropout.h"
#include "mlfe/operators/dropout.h"
#include <algorithm>

namespace mlfe{
namespace module{
namespace layers{

dropout::dropout(float drop_ratio, std::string name)
	: layer_impl<dropout>(name)
{
	__drop_ratio_val = drop_ratio;
}

void dropout::build(std::vector<int> input_shape)
{
	__drop_ratio = add_variable(
		"drop_ratio",
		{1},
		false);
}

Tensor dropout::call(Tensor input)
{
	using namespace mlfe::functional;
	__drop_ratio.mutable_data<float>()[0] = __drop_ratio_val;
	return functional::dropout(input, __drop_ratio);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
