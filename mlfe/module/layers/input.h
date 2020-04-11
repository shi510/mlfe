#pragma once
#include "mlfe/module/layers/layer.h"
#include <vector>
#include <string>

namespace mlfe{
namespace module{
namespace layers{

class input final : public layer_impl<input>
{
public:
	input(std::vector<int> shape, std::string name = "input");

	void build();

	Tensor call();

private:
	std::vector<int> _shape;
	Tensor _in;
};

} // end namespace layer
} // end namespace module
} // end namespace mlfe