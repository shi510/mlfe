#include "graph.h"
#include "mlfe/core/tensor.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <iostream>
#include <unordered_map>

namespace mlfe{

struct node::pimpl
{
    // unique name
    std::string name;
    // task.
    task t;
    // nodes as input.
    std::vector<node> inputs;
    // nodes as output.
    std::vector<node> outputs;
    // node attributes.
    attribute attrs;
    // execution order.
    int order{ 0 };
    // graph.
    std::shared_ptr<graph> g;

    // search node by self key.
    std::unordered_map<node, node> inputs_v2;
    std::unordered_map<node, node> outputs_v2;
};

node::node() : __pimpl(std::make_shared<pimpl>())
{
    __pimpl->g = get_default_graph();
    __pimpl->attrs.add("op_name", std::string("None"));
    __pimpl->name = __pimpl->g->make_unique_name("node");
}

bool node::operator==(const node& n) const
{
    return __pimpl->name == n.__pimpl->name;
}

bool node::operator!=(const node& n) const
{
    return __pimpl->name != n.__pimpl->name;
}

void node::set_name(std::string name)
{
    __pimpl->name = __pimpl->g->make_unique_name(name);
}

std::string node::get_name() const
{
    return __pimpl->name;
}

int node::get_order() const
{
    return __pimpl->order;
}

std::shared_ptr<graph> node::get_graph() const
{
    return __pimpl->g;
}

void node::set_task(const task t)
{
    __pimpl->t.set_task(t);
}

void node::add_input(node& n)
{
    __pimpl->inputs.push_back(n);
    n.__pimpl->outputs.push_back(*this);
    n.__pimpl->t.precede(__pimpl->t);
    if(n.__pimpl->order >= __pimpl->order)
    {
        __pimpl->order = n.__pimpl->order + 1;
    }
}

void node::add_output(node& n)
{
    __pimpl->outputs.push_back(n);
    n.__pimpl->inputs.push_back(*this);
    __pimpl->t.precede(n.__pimpl->t);
}

void node::add_input_v2(node& n)
{
    __pimpl->inputs_v2[n] = n;
    if(n.__pimpl->order >= __pimpl->order)
    {
        __pimpl->order = n.__pimpl->order + 1;
    }
}

void node::add_output_v2(node& n)
{
    n.__pimpl->inputs_v2[*this] = *this;
}

std::vector<node>& node::get_inputs() const
{
    return __pimpl->inputs;
}

std::vector<node> node::get_inputs_v2() const
{
    std::vector<node> vals;
    vals.reserve(__pimpl->inputs_v2.size());

    for(auto kv : __pimpl->inputs_v2) {
        vals.push_back(kv.second);  
    }
    return vals;
}

node node::get_input(int idx) const
{
    if(idx >= __pimpl->inputs.size())
    {
        std::cerr << "node: idx >= __c.size()" << std::endl;
        return node();
    }
    return __pimpl->inputs[idx];
}

std::vector<node>& node::get_outputs() const
{
    return __pimpl->outputs;
}

node node::get_output(int idx) const
{
    if(idx >= __pimpl->outputs.size())
    {
        std::cerr << "node: idx >= __p.size()" << std::endl;
        return node();
    }
    return __pimpl->outputs[idx];
}

void node::add_attr(std::string key, attribute::item it)
{
    __pimpl->attrs.add(key, it);
}

attribute::item node::get_attr(std::string key)
{
    return __pimpl->attrs.get(key);
}

bool node::has_attr(std::string key) const
{
    return __pimpl->attrs.has(key);
}

bool node::has_task() const
{
    return !__pimpl->t.empty();
}

void node::run()
{
    __pimpl->t.run();
}

void node::run_only_mutated()
{
    __pimpl->t.run_only_mutated();
}

void node::run_without_dependencies()
{
    __pimpl->t.run_without_dependencies();
}

bool node::is_mutated() const
{
    return __pimpl->t.is_mutated();
}

void node::set_mutation(bool m)
{
    __pimpl->t.set_mutation(m);
}

std::string node::dump() const
{
    std::stringstream ss;
    for(auto& pp : topological_sort(*this))
    {
        ss << "Node: " << pp.get_name() << std::endl;
        ss << "Op Name: " << *pp.get_attr("op_name").data<std::string>() << std::endl;
        for(auto cc : pp.get_inputs())
        {
            ss << "\tInput: " << cc.get_name() << ", " << cc.is_mutated() << std::endl;
            ss << "\t\tOp Name: " << *cc.get_attr("op_name").data<std::string>() << std::endl;
        }
        for(auto cc : pp.get_outputs())
        {
            ss << "\tOutput: " << cc.get_name() << ", " << cc.is_mutated() << std::endl;
            ss << "\t\tOp Name: " << *cc.get_attr("op_name").data<std::string>() << std::endl;
        }
    }
    return ss.str();
}

std::string node::dump_v2() const
{
    std::stringstream ss;
    for(auto& pp : topological_sort(*this))
    {
        ss << "Node: " << pp.get_name() << " [ ";
        auto shape = pp.get_attr("tensor").data<Tensor>()->shape();
        for(int n = 0; n < shape.size(); ++n){
            ss<<shape[n]<<" ";
        }
        ss<<"]"<<std::endl;
        ss << "Op Name: " << *pp.get_attr("op_name").data<std::string>() << std::endl;
        for(auto cc : pp.__pimpl->inputs_v2)
        {
            auto shape = cc.second.get_attr("tensor").data<Tensor>()->shape();
            ss << "\tInput: " << cc.second.get_name() << " [ ";
            for(int n = 0; n < shape.size(); ++n){
                ss<<shape[n]<<" ";
            }
            ss<<"]"<<std::endl;
            ss << "\t\tOp Name: " << *cc.second.get_attr("op_name").data<std::string>() << std::endl;
        }
        for(auto cc : pp.__pimpl->outputs_v2)
        {
            auto shape = cc.second.get_attr("tensor").data<Tensor>()->shape();
            ss << "\tOutput: " << cc.second.get_name() << " [ ";
            for(int n = 0; n < shape.size(); ++n){
                ss<<shape[n]<<" ";
            }
            ss<<"]"<<std::endl;
            ss << "\t\tOp Name: " << *cc.second.get_attr("op_name").data<std::string>() << std::endl;
        }
    }
    return ss.str();
}

//
// end of implementation of node class.
//

void graph::set_training(const bool training)
{
    __training = training;
}

bool graph::training() const
{
    return __training;
}

std::string graph::make_unique_name(std::string name)
{
    if(__name_idx.find(name) == __name_idx.end())
    {
        __name_idx[name] = 0;
        return name + ":" + std::to_string(__name_idx[name]);
    }
    __name_idx[name] += 1;
    return name + ":" + std::to_string(__name_idx[name]);
}

std::shared_ptr<graph> get_default_graph()
{
    static std::shared_ptr<graph> g0 = std::make_shared<graph>();
    return g0;
}

std::vector<node> topological_sort(const node& r, bool reverse)
{
    std::queue<node> will_visit;
    std::vector<node> visit_list;
    std::unordered_map<node, int> visited;

    will_visit.push(r);
    while(!will_visit.empty()){
        auto v = will_visit.front();
        for(auto &c : v.get_inputs_v2()){
            will_visit.push(c);
        }
        auto it = visited.find(v);
        if(it == visited.end()){
            visit_list.push_back(v);
            visited[v] = 0; // integer zero is dummpy data.
        }
        will_visit.pop();
    }

    // sort by execution order.
    std::sort(visit_list.begin(), visit_list.end(), [](node a, node b) {
        return a.get_order() > b.get_order();
    });

    if(!reverse){
        std::reverse(visit_list.begin(), visit_list.end());
    }
    
    return visit_list;
}


} // end namespace mlfe

namespace std{

size_t hash<mlfe::node>::operator()(const mlfe::node &n) const{
    return hash<shared_ptr<mlfe::node::pimpl>>{}(n.__pimpl);
}

} // end namespace std
