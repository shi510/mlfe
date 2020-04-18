#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class maxpool2d final : public layer_impl<maxpool2d>
{
public:
	maxpool2d(int kernel,
		int stride,
		int padding,
		std::string name = "maxpool2d");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
	int __kernel;
	int __stride;
	int __padding;
	Tensor __w, __b;
	std::mt19937 __rng;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
