#include <iostream>
#include <mlfe/operators.h>
#include <mlfe/core/tensor.h>
#include <mlfe/optimizers.h>
#include <mlfe/core/graph.h>
#include <algorithm>
#include "dataset/cifar.h"
#include "net_models.h"

using namespace mlfe;
using namespace mlfe::module;

// custom metric function.
float categorical_accuracy(Tensor y_true, Tensor y_pred);

template <int _BatchSize>
void train_convnet(
	dataset::cifar10_gen<_BatchSize> train_set,
	dataset::cifar10_gen<_BatchSize> valid_set);


int main(int argc, char *argv[])
{
	std::vector<uint8_t> train_x;
	std::vector<uint8_t> train_y;
	std::vector<uint8_t> valid_x;
	std::vector<uint8_t> valid_y;

	if(argc < 3)
	{
		std::cout<<argv[0];
		std::cout << " [cifar10]";
		std::cout<<" [cifar dataset folder]"<<std::endl;
		return 1;
	}
	// read all data from original cifar10 binary file.
	dataset::read_cifar10_dataset(argv[2],
		train_x, train_y,
		valid_x, valid_y);

	if (std::string(argv[1]) == "cifar10")
	{
		dataset::cifar10_gen<64> train_set(train_x, train_y), valid_set(valid_x, valid_y);
		train_convnet(train_set, valid_set);
	}
	else
	{
		std::cout<<"Wrong command, ";
		std::cout<<"select one of the commands below."<<std::endl;
		std::cout<<" - classifiar"<<std::endl;
	}

	return 0;
}

template <int _BatchSize>
void train_convnet(
	dataset::cifar10_gen<_BatchSize> train_set,
	dataset::cifar10_gen<_BatchSize> valid_set)
{
	auto net = models::conv_net({_BatchSize, 3, 32, 32});
	auto optm = functional::create_gradient_descent_optimizer(5e-3, 0.9);
	auto loss = functional::softmax_cross_entropy;
	net.compile(optm, loss, categorical_accuracy);
	net.fit(train_set, valid_set, 100, _BatchSize);
}


float categorical_accuracy(Tensor y_true, Tensor y_pred)
{
	const int batch_size = y_true.shape()[0];
	const int classes = y_true.shape()[1];
	int correct = 0;
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
			correct += 1;
		}
	}
	return float(correct) / float(batch_size);
}