#include "mlfe/core/node.h"
#include <iostream>

namespace mlfe{

struct node::pimpl
{
	// task.
	task t;
	// nodes as input.
	std::vector<node> inputs;
	// nodes as output.
	std::vector<node> outputs;
};

node::node() : __pimpl(std::make_shared<pimpl>()){}

void node::set_task(const task t)
{
	__pimpl->t = t;
}

void node::add_input(node& n)
{
	__pimpl->inputs.push_back(n);
	n.__pimpl->outputs.push_back(*this);
	n.__pimpl->t.precede(__pimpl->t);
}

void node::add_output(node& n)
{
	__pimpl->outputs.push_back(n);
	n.__pimpl->inputs.push_back(*this);
	__pimpl->t.precede(n.__pimpl->t);
}

const std::vector<node>& node::get_inputs() const
{
	return __pimpl->inputs;
}

node node::get_input(int idx) const
{
	if(idx >= __pimpl->inputs.size())
	{
		std::cerr<<"node: idx >= __c.size()"<<std::endl;
		return node();
	}
	return __pimpl->inputs[idx];
}

const std::vector<node>& node::get_outputs() const
{
	return __pimpl->outputs;
}

node node::get_output(int idx) const
{
	if(idx >= __pimpl->outputs.size())
	{
		std::cerr<<"node: idx >= __p.size()"<<std::endl;
		return node();
	}
	return __pimpl->outputs[idx];
}

bool node::has_task() const
{
	return !__pimpl->t.empty();
}

void node::run() const
{
	__pimpl->t.run();
}

} // end namespace mlfe