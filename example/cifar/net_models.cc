#include "net_models.h"
#include <mlfe/module/module.h>

namespace models
{

using namespace mlfe;
using namespace mlfe::module;
using namespace mlfe::module::layers;

Tensor conv_relu_dropout_block(Tensor in, int channels, float drop_ratio)
{
	Tensor out = conv2d(channels, 3, 1, true)(in);
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

model conv_dropout_net(std::vector<int> input_shape)
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

Tensor conv_bn_relu_block(Tensor in, int channels)
{
	Tensor out = conv2d(channels, 3, 1, true, false)(in);
	out = batch_norm()(out);
	out = relu()(out);
	return out;
}

Tensor dense_bn_relu_block(Tensor in, int out_neurons)
{
	Tensor out = dense(out_neurons, false)(in);
	out = batch_norm()(out);
	out = relu()(out);
	return out;
}

model conv_bn_net(std::vector<int> input_shape)
{
	auto in = input(input_shape)();
	auto out = conv_bn_relu_block(in, 128);
	out = conv_bn_relu_block(out, 128);
	out = maxpool2d(2, 2, 0)(out);
	out = conv_bn_relu_block(out, 256);
	out = conv_bn_relu_block(out, 256);
	out = maxpool2d(2, 2, 0)(out);
	out = conv_bn_relu_block(out, 512);
	out = conv_bn_relu_block(out, 512);
	out = maxpool2d(2, 2, 0)(out);
	out = flatten()(out);
	out = dense_bn_relu_block(out, 1024);
	out = dense(10)(out);
	return model(in, out);
}

} // end namespace models