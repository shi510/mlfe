#include "mlfe/module/model.h"

namespace mlfe
{
namespace module
{

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
