#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class dropout final : public layer_impl<dropout>
{
public:
	dropout(float drop_ratio, std::string name = "dropout");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
	float __drop_ratio_val;
	Tensor __drop_ratio;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
