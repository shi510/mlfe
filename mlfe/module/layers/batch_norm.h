#pragma once
#include "mlfe/module/layers/layer.h"
#include <random>

namespace mlfe{
namespace module{
namespace layers{

class batch_norm final : public layer_impl<batch_norm>
{
public:
	batch_norm(std::string name = "batch_norm");

	void build(std::vector<int> input_shape);

	Tensor call(Tensor input);

private:
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe
