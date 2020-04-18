#pragma once
#include <vector>
#include <mlfe/module/module.h>
#include <iostream>

namespace models
{

using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

auto conv_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = conv2d(16, 5, 1, 2)(in);
	out = maxpool2d(2, 2, 0)(out);
	out = relu()(out);
	out = conv2d(24, 5, 1, 2)(out);
	out = maxpool2d(2, 2, 0)(out);
	out = relu()(out);
	out = flatten()(out);
	out = dense(128)(out);
	out = relu()(out);
	out = dense(10)(out);
	return model(in, out);
}

} // end namespace models