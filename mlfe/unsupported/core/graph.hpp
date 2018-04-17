#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__
#include <vector>
#include <memory>
#include <map>
#include "node.hpp"

namespace mlfe{
class Graph final{
public:
    Graph(Workspace *ws, Accelerator acc = Accelerator::Default, 
        DataType data_type = DataType::F32);

    Graph(const Graph &) = default;

    Graph &operator=(const Graph &) = default;

    void Init(bool with_gradient = false) const;

    void Run() const;

    void Gradient();

    std::vector<std::shared_ptr<const node::Node>> Nodes() const;

    DataType Type() const;

    Accelerator Accel() const;

    template <typename T,
        typename = typename std::enable_if<std::is_base_of<node::Node, T>::value, T>::type
    >
    void Add(T &node) {
        AddNode(std::make_shared<T>(node));
    }

protected:
    void AddNode(std::shared_ptr<node::Node> node);

private:
    typedef struct InternalGraph;
    std::shared_ptr<InternalGraph> _igraph;
};

} // end namespace mlfe
#endif // end ifndef __GRAPH_HPP__
