#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__
#include "variable.h"
#include <string>
#include <vector>
#include <memory>

namespace mlfe{
class OpDependency;

class Tensor final : public Variable{
public:
    class DependencyAdder;

    Tensor();

    Tensor(std::string name);

    explicit Tensor(std::vector<int> shape);

    ~Tensor();

    Tensor(const Tensor &t) = default;

    Tensor &operator=(DependencyAdder dep_adder);

    void Initialize(Tensor init);

    OpDependency InitDependency() const;

    std::string info() const;

    bool operator==(const Tensor &v) const;

    void add_parent(const Tensor p);

    void add_child(const Tensor c);

    std::vector<Tensor> get_parents() const;

    std::vector<Tensor> get_children() const;

    int get_exec_order() const;

    OpDependency get_dep() const;

    void set_trainable(bool trainable);

    bool get_trainable() const;

private:
    friend struct std::hash<Tensor>;
    struct InternalData;
    std::shared_ptr<InternalData> internal_data;
};

class Tensor::DependencyAdder{
public:
    DependencyAdder(OpDependency dep);

private:
    friend class Tensor;
    struct pimpl;
    std::shared_ptr<pimpl> _impl;
};

} // end namespace mlfe

namespace std{

template <>
struct hash<mlfe::Tensor>{
    size_t operator()(const mlfe::Tensor &v) const;
};

} // end namespace std
#endif // end ifndef __TENSOR_HPP__
