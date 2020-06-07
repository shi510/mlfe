#include "net_models.h"

namespace models
{
using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

model simple_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = dense(10)(in);
	return model(in, out);
}

model conv_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = conv2d(16, 5, 1, true)(in);
	out = maxpool2d(2, 2, 0)(out);
	out = relu()(out);
	out = conv2d(24, 5, 1, true)(out);
	out = maxpool2d(2, 2, 0)(out);
	out = relu()(out);
	out = flatten()(out);
	out = dense(128)(out);
	out = relu()(out);
	out = dense(10)(out);
	return model(in, out);
}

model auto_encoder(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto encoder = [](Tensor in)
		{
			auto enc = dense(300)(in);
			enc = sigmoid()(enc);
			enc = dense(150)(enc);
			enc = sigmoid()(enc);
			enc = dense(50)(enc);
			enc = sigmoid()(enc);
			enc = dense(10)(enc);
			enc = sigmoid()(enc);
			return enc;
		};
	auto decoder = [&in](Tensor enc, int out_size)
		{
			auto dec = dense(50)(enc);
			dec = sigmoid()(dec);
			dec = dense(150)(dec);
			dec = sigmoid()(dec);
			dec = dense(300)(dec);
			dec = sigmoid()(dec);
			dec = dense(out_size)(dec);
			dec = sigmoid()(dec);
			return model(in, dec);
		};
	
	return decoder(encoder(in), input_shape[0]);
}

} // end namespace models