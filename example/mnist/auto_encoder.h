#pragma once
#include <vector>
#include <mlfe/module/module.h>

namespace models
{

using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

auto auto_encoder(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto encoder = [](Tensor in)
		{
			auto enc = dense(300)(in);
			enc = relu()(enc);
			enc = dense(150)(enc);
			enc = relu()(enc);
			enc = dense(50)(enc);
			enc = relu()(enc);
			enc = dense(10)(enc);	
			return enc;
		};
	auto decoder = [&in](Tensor enc, int out_size)
		{
			auto dec = dense(50)(enc);
			dec = relu()(dec);
			dec = dense(50)(dec);
			dec = relu()(dec);
			dec = dense(150)(dec);
			dec = relu()(dec);
			dec = dense(300)(dec);
			dec = relu()(dec);
			dec = dense(out_size)(dec);
			// dec = sigmoid()(dec);
			return model(in, dec);
		};
	
	return decoder(encoder(in), input_shape[1]);
}

} // end namespace models