#pragma once
#include "mlfe/module/layers/layer.h"

namespace mlfe{
namespace module{
namespace layers{

class sigmoid final : public layer_impl<sigmoid>
{
public:
	sigmoid(std::string name = "sigmoid");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
