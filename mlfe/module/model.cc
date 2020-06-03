#include "mlfe/module/model.h"
#include "mlfe/core/op_algo.h"
#include <algorithm>

namespace mlfe
{
namespace module
{

model::model(Tensor input, Tensor output, std::string name)
{
	__input = input;
	__output = output;
	__true = functional::create_variable(__output.shape());
	__name = name;
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

	// make computation graph for gradient nodes.
	__loss.backprop();
	// collect all nodes related to trainable variables.
	auto topo_list = topological_sort(__loss.get_node());
	for (auto back_node : topo_list)
	{
		if(back_node.has_attr("trainable"))
		{
			if(*back_node.get_attr("trainable").data<bool>())
			{
				auto train_v = back_node.get_attr("tensor").data<Tensor>();
				auto list = topological_sort(train_v->get_backprop_node());
				__train_seq.insert(__train_seq.end(), list.begin(), list.end());
				__train_vars.push_back(*train_v);
			}
		}
	}
	// remove duplicated nodes.
	std::sort(__train_seq.begin(), __train_seq.end(), [](node a, node b) {
		return a.get_name() > b.get_name();
	});
	__train_seq.erase(std::unique(__train_seq.begin(), __train_seq.end()), __train_seq.end());
	std::sort(__train_seq.begin(), __train_seq.end(), [](node a, node b) {
		return a.get_order() < b.get_order();
	});

	// make forward sequance.
	__fwd_seq = topo_list;
	std::sort(__fwd_seq.begin(), __fwd_seq.end(), [](node a, node b) {
		return a.get_name() > b.get_name();
	});
	__fwd_seq.erase(std::unique(__fwd_seq.begin(), __fwd_seq.end()), __fwd_seq.end());
	std::sort(__fwd_seq.begin(), __fwd_seq.end(), [](node a, node b) {
		return a.get_order() < b.get_order();
	});
}

Tensor model::get_input() const
{
	return __input;
}

Tensor model::get_output() const
{
	return __output;
}

std::string model::get_name() const
{
	return __name;
}

void model::resize(int batch_size)
{
	if (__input.shape()[0] != batch_size)
	{
		std::vector<int> resize_shape = __input.shape();
		resize_shape[0] = batch_size;
		__input.resize(resize_shape);
		resize_shape = __output.shape();
		resize_shape[0] = batch_size;
		__output.resize(resize_shape);
		__true.resize(resize_shape);
		for(auto& nd : __fwd_seq){
			auto op_name = *nd.get_attr("op_name").data<std::string>();
			auto op = *nd.get_attr("op").data<std::shared_ptr<OpAlgo>>();
			auto t = *nd.get_attr("tensor").data<Tensor>();
			op->resize();
		}
	}
}

} // end namespace module
} // end namespace mlfe
