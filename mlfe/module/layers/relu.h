#pragma once
#include "mlfe/module/layers/layer.h"

namespace mlfe{
namespace module{
namespace layers{

class relu final : public layer_impl<relu>
{
public:
	relu(std::string name = "relu");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
