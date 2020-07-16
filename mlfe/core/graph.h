#pragma once
#include "mlfe/core/task.h"
#include "mlfe/core/attribute.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace mlfe{

// forward declaration.
class graph;

class node
{
public:
    node();

    bool operator==(const node& n) const;

    bool operator!=(const node& n) const;

    void set_name(std::string name);

    std::string get_name() const;

    int get_order() const;

    std::shared_ptr<graph> get_graph() const;

    void set_task(const task t);

    void add_input(node& n);

    void add_output(node& n);

    std::vector<node>& get_inputs() const;

    node get_input(int idx) const;

    std::vector<node>& get_outputs() const;

    node get_output(int idx) const;

    void add_attr(std::string key, attribute::item it);

    attribute::item get_attr(std::string key);

    bool has_attr(std::string key) const;

    bool has_task() const;

    void run();

    void run_only_mutated();

    void run_without_dependencies();

    bool is_mutated() const;

    void set_mutation(bool m);

    std::string dump() const;

private:
    struct pimpl;
    std::shared_ptr<pimpl> __pimpl;
};

//
// end of declaration of node class
//

std::vector<node> topological_sort(const node& r);

class graph
{
public:
	void set_training(const bool training);

	bool training() const;

    std::string make_unique_name(std::string name);

private:
	bool __training;
    std::map<std::string, node> __unique;
    std::map<std::string, int> __name_idx;
};

std::shared_ptr<graph> get_default_graph();

} // end namespace mlfe
