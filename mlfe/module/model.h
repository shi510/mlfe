#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/graph.h"
#include "mlfe/optimizers/optimizer.h"
#include "mlfe/operators/math.h"
#include <functional>
#include <string>
#include <iostream>

namespace mlfe
{
namespace module
{

class model
{
	typedef Tensor Y_Pred;
	typedef Tensor Y_True;
	typedef std::function<Tensor(Y_True, Y_Pred)> LossFn;
	typedef std::function<float(Y_True, Y_Pred)> MetricFn;

public:
	model(Tensor input, Tensor output, std::string name = "model");

	void compile(std::shared_ptr<opt::optimizer> optim,
				 LossFn loss_fn,
				 MetricFn metric_fn = nullptr);

	template <typename T>
	std::vector<T> operator()(std::vector<T> x, std::vector<int> shape)
	{
		if (x.size() != __input.size() || shape.size() != __input.dims())
		{
			std::cout << "x.size() != __input.size() || ";
			std::cout << "shape.size() != __input.dims()" << std::endl;
			return {};
		}
		
		std::vector<T> y(__output.size());
		std::copy(x.begin(), x.end(), __input.begin<T>());
		__output.eval();
		std::copy(__output.begin<T>(), __output.end<T>(), y.begin());
		return y;
	}

	template <typename _Callable>
	void fit(
		_Callable train_set,
		_Callable valid_set,
		const int epoch,
		const int batch_size)
	{
		typedef typename std::remove_reference<
			decltype(std::get<0>(train_set(0))[0])>::type _TypeX;
		typedef typename std::remove_reference<
			decltype(std::get<1>(train_set(0))[0])>::type _TypeY;

		for (int n = 0; n < epoch; ++n)
		{
			std::cout << "epoch : " << n << std::endl;
			auto [train_loss, train_acc] = __iter<_Callable, _TypeX, _TypeY>(
				train_set, train_set.size() / batch_size, true);
			auto [valid_loss, valid_acc] = __iter<_Callable, _TypeX, _TypeY>(
				valid_set, valid_set.size() / batch_size, false);
			std::cout << "train loss : " << train_loss << ", ";
			std::cout << "valid loss : " << valid_loss << std::endl;
			if(__metric_fn)
			{
				std::cout << "train accuracy : " << train_acc << ", ";
				std::cout << "valid accuracy : " << valid_acc << std::endl;
			}
		}
	}

private:
	template <typename _Callable, typename _TypeX, typename _TypeY>
	std::tuple<float, float> __iter(_Callable &data_set, const int iter, const bool train)
	{
		float total_loss = 0.f;
		float accuracy = 0.f;
		for (int n = 0; n < iter; ++n)
		{
			auto [x, y] = data_set(n);
			std::copy(x.begin(), x.end(), __input.begin<_TypeX>());
			std::copy(y.begin(), y.end(), __true.begin<_TypeY>());
			__loss.eval();
			if (train)
			{
				__loss.backprop();
				for (auto &var : __train_vars)
				{
					__optim->apply(var, var.grad());
				}
			}
			total_loss += __loss.data<float>()[0];
			if(__metric_fn)
			{
				accuracy += __metric_fn(__true, __output);
			}
		}
		return std::make_tuple(total_loss / iter, accuracy / iter);
	}

private:
	Tensor __input;
	Tensor __output;
	Tensor __loss;
	Tensor __true;
	LossFn __loss_fn;
	MetricFn __metric_fn;
	std::vector<Tensor> __train_vars;
	std::shared_ptr<opt::optimizer> __optim;
};

model::model(Tensor input, Tensor output, std::string name)
{
	__input = input;
	__output = output;
	__true = functional::create_variable(__output.shape());
}

void model::compile(
	std::shared_ptr<opt::optimizer> optim,
	LossFn loss_fn,
	MetricFn metric_fn)
{
	__loss_fn = loss_fn;
	__optim = optim;
	__metric_fn = metric_fn;
	__loss = functional::mean(__loss_fn(__output, __true));
	__loss.set_name("loss");
	__output.set_name("output");
	__true.set_name("true");
	for (auto& var : visit_bfs(__loss))
	{
		if (var.trainable())
		{
			__train_vars.push_back(var);
		}
	}
}

} // end namespace module
} // end namespace mlfe
