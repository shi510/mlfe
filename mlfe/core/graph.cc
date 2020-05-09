#include "graph.h"
#include <queue>

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

std::vector<node> topological_sort(const node& r)
{
    std::queue<node> will_visit;
    std::vector<node> visit_list;

    will_visit.push(r);
    while(!will_visit.empty()){
        auto v = will_visit.front();
        for(auto &c : v.get_inputs()){
            will_visit.push(c);
        }
        visit_list.push_back(v);
        will_visit.pop();
    }
    return visit_list;
}


} // end namespace mlfe
