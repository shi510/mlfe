#ifndef __GRAPH_H__
#define __GRAPH_H__
#include <vector>

namespace mlfe{
// forward declaration.
class Tensor;

std::vector<Tensor> visit_bfs(const Tensor root);

// Graph is only for saving node's infomation.
// TODO:
//   1. grant unique name to each node.
//   2. save node info.
//   3. save node value.
class Graph{
public:
    
private:
    
};

} // end namespace mlfe

#endif // end #ifndef __GRAPH_H__
