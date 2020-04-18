#include "mlfe/module/layers/maxpool2d.h"
#include "mlfe/operators/basic_arithmetics.h"
#include "mlfe/operators/broadcasting.h"
#include "mlfe/operators/convolution.h"
#include "mlfe/operators/pool.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace module{
namespace layers{

namespace fn = functional;

maxpool2d::maxpool2d(int kernel,
	int stride,
	int padding,
	std::string name)
	: layer_impl<maxpool2d>(name)
{
	__kernel = kernel;
	__stride = stride;
	__padding = padding;
}

void maxpool2d::build(std::vector<int> input_shape)
{
}

Tensor maxpool2d::call(Tensor input)
{
	return fn::pool_max(input, {__kernel, __kernel}, {__stride, __stride}, {__padding, __padding});
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
