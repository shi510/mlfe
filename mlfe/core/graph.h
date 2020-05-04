#ifndef __GRAPH_H__
#define __GRAPH_H__
#include <vector>
#include <memory>

namespace mlfe{
// forward declaration.
class Tensor;

std::vector<Tensor> visit_bfs(const Tensor root);

class graph{
public:
	void set_training(const bool training);

	bool training() const;

private:
	bool __training;
};

std::shared_ptr<graph> get_default_graph();

} // end namespace mlfe

#endif // end #ifndef __GRAPH_H__
