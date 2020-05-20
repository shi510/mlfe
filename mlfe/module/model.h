#pragma once
#include "mlfe/core/tensor.h"
#include "mlfe/core/graph.h"
#include "mlfe/optimizers/optimizer.h"
#include "mlfe/operators/math.h"
#include "mlfe/module/callbacks/callback.h"
#include "mlfe/utils/templates.h"
#include <functional>
#include <string>
#include <map>
#include <iostream>
#include <algorithm>

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

	Tensor get_input() const;

	Tensor get_output() const;

	std::string get_name() const;

	template <typename T>
	std::vector<T> operator()(std::vector<T> x, std::vector<int> shape)
	{
		if(x.size() != __input.size() || shape.size() != __input.dims())
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
		const int batch_size,
		util::template_unpacker<callback> callbacks = {})
	{
		typedef typename std::remove_reference<
			decltype(std::get<0>(train_set(0))[0])>::type _TypeX;
		typedef typename std::remove_reference<
			decltype(std::get<1>(train_set(0))[0])>::type _TypeY;
		std::map<std::string, float> logs;
		std::for_each(callbacks.params.begin(), callbacks.params.end(),
			[this](std::shared_ptr<callback> cb) {cb->set_model(this); });
		for(int n = 0; n < epoch; ++n)
		{
			std::cout << "epoch : " << n << std::endl;
			std::for_each(callbacks.params.begin(), callbacks.params.end(),
				[&logs, &n](std::shared_ptr<callback> cb) {cb->on_train_begin(n, logs); });
			auto [train_loss, train_acc] = __iter<_Callable, _TypeX, _TypeY>(
				train_set, train_set.size() / batch_size, true);
			std::for_each(callbacks.params.begin(), callbacks.params.end(),
				[&logs, &n](std::shared_ptr<callback> cb) {cb->on_train_end(n, logs); });
			if(__metric_fn)
			{
				logs["train/accuracy"] = train_acc;
			}
			logs["train/loss"] = train_loss;

			std::for_each(callbacks.params.begin(), callbacks.params.end(),
				[&logs, &n](std::shared_ptr<callback> cb) {cb->on_test_begin(n, logs); });
			auto [valid_loss, valid_acc] = __iter<_Callable, _TypeX, _TypeY>(
				valid_set, valid_set.size() / batch_size, false);
			std::for_each(callbacks.params.begin(), callbacks.params.end(),
				[&logs, &n](std::shared_ptr<callback> cb) {cb->on_test_end(n, logs); });
			if(__metric_fn)
			{
				logs["valid/accuracy"] = valid_acc;
			}
			logs["valid/loss"] = valid_loss;
			logs["vars/learning_rate"] = __optim->get_learning_rate();
			std::for_each(callbacks.params.begin(), callbacks.params.end(),
				[&logs, &n](std::shared_ptr<callback> cb) {cb->on_epoch_end(n, logs); });
			std::cout << "train loss : " << train_loss << ", ";
			std::cout << "valid loss : " << valid_loss << std::endl;
			if(__metric_fn)
			{
				std::cout << "train accuracy : " << train_acc << ", ";
				std::cout << "valid accuracy : " << valid_acc << std::endl;
			}
		}
	}

	std::shared_ptr<opt::optimizer> get_optimizer() const
	{
		return __optim;
	}

	std::vector<Tensor> get_train_variables() const
	{
		return __train_vars;
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
			__loss.get_graph()->set_training(train);
			if(!train)
			{
				__loss.eval();
			}
			else
			{
				for_each(__train_seq.begin(), __train_seq.end(), [](node n) {
					n.run_without_dependencies();
					});
				for(auto& var : __train_vars)
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
	std::string __name;
	Tensor __input;
	Tensor __output;
	Tensor __loss;
	Tensor __true;
	LossFn __loss_fn;
	MetricFn __metric_fn;
	std::vector<node> __train_seq;
	std::vector<Tensor> __train_vars;
	std::shared_ptr<opt::optimizer> __optim;
};

} // end namespace module
} // end namespace mlfe
