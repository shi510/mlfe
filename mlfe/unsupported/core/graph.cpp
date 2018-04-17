#include "graph.hpp"

namespace mlfe{
struct Graph::InternalGraph {
public:
    std::vector<std::shared_ptr<node::Node>> _nodes;
    std::shared_ptr<node::Node> _current_node;
    Workspace *_ws;
    DataType _data_type;
    Accelerator _acc;
};

Graph::Graph(Workspace *ws, Accelerator acc, 
    DataType data_type) {
    _igraph = std::make_shared<InternalGraph>();
    _igraph->_data_type = data_type;
    _igraph->_acc = acc;
    _igraph->_ws = ws;
    _igraph->_current_node = nullptr;
}

void Graph::Init(bool with_gradient) const{
    if (with_gradient) {
        for (int n = _igraph->_nodes.size() - 1; n >= 0; --n) {
            node::NodeGradient *node_grad =
                reinterpret_cast<node::NodeGradient * > (_igraph->_nodes[n].get());

            node_grad->Init(_igraph->_ws);
        }
    }
}

void Graph::Run() const{
    for (auto &node : _igraph->_nodes) {
        node->Run();
    }
}

void Graph::Gradient() {
    for (int n = _igraph->_nodes.size() - 1; n >= 0; --n) {
        node::NodeGradient *node_grad = 
            reinterpret_cast<node::NodeGradient * > (_igraph->_nodes[n].get());
        
        node_grad->Run();
    }
}

std::vector<std::shared_ptr<const node::Node>> Graph::Nodes() const {
    std::vector<std::shared_ptr<const node::Node>> nodes;
    for (auto node : _igraph->_nodes) {
        nodes.push_back(node);
    }
    return nodes;
}

DataType Graph::Type() const {
    return _igraph->_data_type;
}

Accelerator Graph::Accel() const {
    return _igraph->_acc;
}

void Graph::AddNode(std::shared_ptr<node::Node> node) {
    node->Accel(_igraph->_acc);
    node->Type(_igraph->_data_type);
    node->Init(_igraph->_ws);
    _igraph->_nodes.push_back(node);
    _igraph->_current_node = node;
}

} // end namespace mlfe
