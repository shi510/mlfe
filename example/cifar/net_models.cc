#include "net_models.h"
#include <mlfe/module/module.h>

namespace models
{

using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

Tensor conv_relu_dropout_block(Tensor in, int channels, float drop_ratio)
{
	Tensor out = conv2d(channels, 3, 1, 1)(in);
	out = relu()(out);
	out = dropout(drop_ratio)(out);
	return out;
}

Tensor dense_relu_dropout_block(Tensor in, int out_neurons, float drop_ratio)
{
	Tensor out = dense(out_neurons)(in);
	out = relu()(out);
	out = dropout(drop_ratio)(out);
	return out;
}

model conv_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = conv_relu_dropout_block(in, 64, 0.3);
	out = maxpool2d(2, 2, 0)(out);
	out = conv_relu_dropout_block(out, 128, 0.3);
	out = maxpool2d(2, 2, 0)(out);
	out = conv_relu_dropout_block(out, 256, 0.3);
	out = maxpool2d(2, 2, 0)(out);
	out = flatten()(out);
	out = dense_relu_dropout_block(out, 512, 0.5);
	out = dense_relu_dropout_block(out, 256, 0.5);
	out = dense(10)(out);
	return model(in, out);
}

} // end namespace models