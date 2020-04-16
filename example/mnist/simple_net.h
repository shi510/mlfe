#pragma once
#include <vector>
#include <mlfe/module/module.h>

namespace models
{

using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

auto simple_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = dense(10)(in);
	return model(in, out);
}

} // end namespace models