#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class flatten final : public layer_impl<flatten>
{
public:
	flatten(std::string name = "flatten");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
	int __out_elements;
	int __batch;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
