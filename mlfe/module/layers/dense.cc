#include "mlfe/module/layers/dense.h"
#include "mlfe/operators/matmul.h"
#include "mlfe/operators/basic_arithmetics.h"
#include <algorithm>

namespace mlfe{
namespace module{
namespace layers{

dense::dense(int out_features,
	bool use_bias,
	std::string name)
	: layer_impl<dense>(name)
{
	_out_features = out_features;
	__use_bias = use_bias;
}

void dense::build(std::vector<int> input_shape)
{
	auto kaiming_he_fn = [&]() {
		float std = std::sqrt(6.f / input_shape[1]);
		auto dist = std::uniform_real_distribution<float>(-std, std);
		return dist(__rng);
	};
	_w = add_variable(
		"weights",
		{input_shape[1], _out_features},
		true);
	std::generate(_w.mutable_data<float>(),
		_w.mutable_data<float>() + _w.size(),
		kaiming_he_fn);
	if(__use_bias)
	{
		_b = add_variable(
			"bias",
			{ _out_features },
			true);
		std::fill(_b.mutable_data<float>(),
			_b.mutable_data<float>() + _b.size(),
			0.1f);
	}
}

Tensor dense::call(Tensor input)
{
	using namespace mlfe::functional;
	auto y = matmul(input, _w);
	if(__use_bias)
	{
		y = add(y, _b);
	}
	return y;
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
