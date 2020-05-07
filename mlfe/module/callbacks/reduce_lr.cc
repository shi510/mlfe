#pragma once
#include "mlfe/module/callbacks/reduce_lr.h"
#include "mlfe/module/model.h"
#include <limits>

namespace mlfe{
namespace module{

reduce_lr::reduce_lr(std::string monitor,
	int patience,
	double factor,
	double min_lr)
{
	__monitor = monitor;
	__patience = patience;
	__factor = factor;
	__min_lr = min_lr;
	__count = 0;
	__best = std::numeric_limits<float>::max();
}

void reduce_lr::on_train_begin(const int epoch,
	const std::map<std::string, float>& logs)
{
	auto it = logs.find(__monitor);
	if(it == logs.end())
	{
		return;
	}
	auto val = it->second;
	if(val < __best)
	{
		__best = val;
		__count = 0;
	}
	else
	{
		__count += 1;
		if(__count >= __patience)
		{
			auto cur_lr = __m->get_optimizer()->get_learning_rate();
			if (cur_lr > __min_lr)
			{
				auto new_lr = cur_lr * __factor;
				new_lr = std::max(new_lr, __min_lr);
				__m->get_optimizer()->update_learning_rate(new_lr);
				__count = 0;
				std::cout << "downscale learning rate from " << cur_lr << " to " << __m->get_optimizer()->get_learning_rate() << std::endl;
			}
		}
	}
}

} // end namespace module
} // end namespace mlfe