#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class conv2d final : public layer_impl<conv2d>
{
public:
	conv2d(int out_channels,
		int kernel,
		int stride,
		int padding,
		bool use_bias = true,
		std::string name = "conv2d");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
	int __out_channels;
	int __kernel;
	int __stride;
	int __padding;
	bool __use_bias;
	Tensor __w, __b;
	std::mt19937 __rng;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
