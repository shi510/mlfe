#include "mlfe/module/layers/dense.h"
#include "mlfe/operators/matmul.h"
#include "mlfe/operators/basic_arithmetics.h"
#include <algorithm>

namespace mlfe{
namespace module{
namespace layers{

dense::dense(int out_features, std::string name)
	: layer_impl<dense>(name)
{
	_out_features = out_features;
}

void dense::build(std::vector<int> input_shape)
{
	auto random_fn = [this](){
		auto dist = std::normal_distribution<float>(0, 0.1);
		return dist(__rng);
	};
	_w = add_variable(
		"weights",
		{input_shape[1], _out_features},
		true);
	_b = add_variable(
		"bias",
		{_out_features},
		true);
	std::fill(_b.mutable_data<float>(),
		_b.mutable_data<float>() + _b.size(),
		0);
	std::generate(_w.mutable_data<float>(),
		_w.mutable_data<float>() + _w.size(),
		random_fn);
}

Tensor dense::call(Tensor input)
{
	using namespace mlfe::functional;
	return add(matmul(input, _w), _b);
}

} // end namespace layer
} // end namespace module
} // end namespace mlfe
