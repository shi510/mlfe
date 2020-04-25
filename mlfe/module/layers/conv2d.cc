#include "mlfe/module/layers/conv2d.h"
#include "mlfe/operators/basic_arithmetics.h"
#include "mlfe/operators/broadcasting.h"
#include "mlfe/operators/convolution.h"
#include <algorithm>
#include <numeric>

namespace mlfe{
namespace module{
namespace layers{

namespace fn = functional;

conv2d::conv2d(int out_channels,
	int kernel,
	int stride,
	int padding,
	std::string name)
	: layer_impl<conv2d>(name)
{
	__out_channels = out_channels;
	__kernel = kernel;
	__stride = stride;
	__padding = padding;
}

void conv2d::build(std::vector<int> input_shape)
{
	int in_elem = std::accumulate(input_shape.begin()+1,
		input_shape.end(), 1, std::multiplies<int>());
	auto kaiming_he_fn = [&]() {
		float std = std::sqrt(6.f / (in_elem));
		auto dist = std::uniform_real_distribution<float>(-std, std);
		return dist(__rng);
	};
	__w = add_variable("weights", { __out_channels, input_shape[1], __kernel, __kernel }, true);
	__b = add_variable("bias", { 1, __out_channels, 1, 1 }, true);
	std::generate(__w.begin<float>(), __w.end<float>(), kaiming_he_fn);
	std::fill(__b.begin<float>(), __b.end<float>(), 0.1f);
}

Tensor conv2d::call(Tensor input)
{
	Tensor y = fn::conv2d(input, __w, {__stride, __stride}, {__padding, __padding});
	return fn::add(y, fn::broadcast(__b, y.shape()));
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
