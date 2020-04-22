#include <iostream>
#include <mlfe/operators.h>
#include <mlfe/core/tensor.h>
#include <mlfe/optimizers.h>
#include "dataset/mnist.h"
#include "net_models.h"

using namespace mlfe;
using namespace mlfe::module;

// custom metric function.
float categorical_accuracy(Tensor y_true, Tensor y_pred);

template <int _BatchSize>
void train_simplenet(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set);

template <int _BatchSize>
void train_convnet(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set);

template <int _BatchSize>
void train_autoencoder(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set);

int main(int argc, char *argv[])
{
	std::vector<uint8_t> train_x;
	std::vector<uint8_t> train_y;
	std::vector<uint8_t> valid_x;
	std::vector<uint8_t> valid_y;

	if(argc < 3)
	{
		std::cout<<argv[0];
		std::cout<<" [simple | conv | autoencoder]";
		std::cout<<" [mnist dataset folder]"<<std::endl;
		return 1;
	}
	// read all data from original mnist binary file.
	dataset::read_mnist_dataset(argv[2],
		train_x, train_y,
		valid_x, valid_y);

	if(std::string(argv[1]) == "simple")
	{
		dataset::mnist_gen<32> train_set(train_x, train_y), valid_set(valid_x, valid_y);
		train_simplenet(train_set, valid_set);
	}
	else if(std::string(argv[1]) == "conv")
	{
		dataset::mnist_gen<64> train_set(train_x, train_y), valid_set(valid_x, valid_y);
		train_convnet(train_set, valid_set);
	}
	else if(std::string(argv[1]) == "autoencoder")
	{
		dataset::mnist_gen<32> train_set(train_x, train_y, true), valid_set(valid_x, valid_y, true);
		train_autoencoder(train_set, valid_set);
	}
	else
	{
		std::cout<<"Wrong command, ";
		std::cout<<"select one of the commands below."<<std::endl;
		std::cout<<" - simple"<<std::endl;
		std::cout<<" - conv"<<std::endl;
		std::cout<<" - autoencoder"<<std::endl;
	}

	return 0;
}

template <int _BatchSize>
void train_simplenet(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set)
{
	auto net = models::simple_net({_BatchSize, 28 * 28});
	auto optm = functional::create_gradient_descent_optimizer(1e-1, 0);
	auto loss = functional::softmax_cross_entropy;
	net.compile(optm, loss, categorical_accuracy);
	net.fit(train_set, valid_set, 2, _BatchSize);
}

template <int _BatchSize>
void train_convnet(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set)
{
	auto net = models::conv_net({_BatchSize, 1, 28, 28});
	auto optm = functional::create_gradient_descent_optimizer(2e-2, 0.9);
	auto loss = functional::softmax_cross_entropy;
	net.compile(optm, loss, categorical_accuracy);
	net.fit(train_set, valid_set, 5, _BatchSize);
}

template <int _BatchSize>
void train_autoencoder(
	dataset::mnist_gen<_BatchSize> train_set,
	dataset::mnist_gen<_BatchSize> valid_set)
{
	auto net = models::auto_encoder({_BatchSize, 28 * 28});
	auto optm = functional::create_gradient_descent_optimizer(1e-2, 0.9);
	auto loss = functional::squared_difference;
	net.compile(optm, loss);
	net.fit(train_set, valid_set, 3, _BatchSize);
}

float categorical_accuracy(Tensor y_true, Tensor y_pred)
{
	const int batch_size = y_true.shape()[0];
	const int classes = y_true.shape()[1];
	float accuracy = 0.f;
	for(int b = 0; b < batch_size; ++b)
	{
		auto y_pred_pos = std::max_element(
			y_pred.cbegin<float>() + b * classes,
			y_pred.cbegin<float>() + (b + 1) * classes);
		int class_id = std::distance(
			y_pred.cbegin<float>() + b * classes,
			y_pred_pos);
		if(y_true.data<float>()[b * classes + class_id] == 1.f)
		{
			accuracy += 1.f;
		}
	}
	return accuracy / batch_size;
}