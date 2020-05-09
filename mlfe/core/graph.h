#pragma once
#include "mlfe/core/node.h"
#include <vector>
#include <memory>

namespace mlfe{

std::vector<node> topological_sort(const node& r);

class graph
{
public:
	void set_training(const bool training);

	bool training() const;

private:
	bool __training;
};

std::shared_ptr<graph> get_default_graph();

} // end namespace mlfe
