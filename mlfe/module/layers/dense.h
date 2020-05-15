#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class dense final : public layer_impl<dense>
{
public:
	dense(int out_features,
		bool use_bias = true,
		std::string name = "dense");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
	int _out_features;
	bool __use_bias;
	Tensor _w, _b;
	std::mt19937 __rng;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
