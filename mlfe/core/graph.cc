#include "graph.h"
#include "op_algo.h"
#include "tensor.h"
#include <queue>
#include <unordered_set>

namespace mlfe{

void graph::set_training(const bool training)
{
    __training = training;
}

bool graph::training() const
{
    return __training;
}

std::shared_ptr<graph> get_default_graph()
{
    static std::shared_ptr<graph> g0 = std::make_shared<graph>();
    return g0;
}

std::vector<Tensor> visit_bfs(const Tensor root){
    std::queue<Tensor> will_visit;
    std::vector<Tensor> visit_list;
    std::unordered_set<Tensor> visited;

    will_visit.push(root);
    while(!will_visit.empty()){
        auto v = will_visit.front();
        if(visited.find(v) == visited.end()){
            visited.insert(v);
            visit_list.push_back(v);
            for(auto &c : v.get_children()){
                will_visit.push(c);
            }
        }
        will_visit.pop();
    }
    return visit_list;
}

} // end namespace mlfe
