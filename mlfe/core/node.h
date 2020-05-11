#pragma once
#include "mlfe/core/task.h"
#include <string>
#include <vector>
#include <queue>
#include <unordered_set>
#include <memory>

namespace mlfe
{

class node
{
public:
	node();

	void set_task(const task t);

	void add_input(node& n);

	void add_output(node& n);

	const std::vector<node>& get_inputs() const;

	node get_input(int idx) const;

	const std::vector<node>& get_outputs() const;

	node get_output(int idx) const;

	bool has_task() const;

	void run();

private:
	struct pimpl;
	std::shared_ptr<pimpl> __pimpl;
};

template <typename T>
std::vector<T> visit_bfs(const T root){
    std::queue<T> will_visit;
    std::vector<T> visit_list;
    std::unordered_set<T> visited;

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