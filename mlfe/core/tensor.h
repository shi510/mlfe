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
    class InitDependencyAdder;

    Tensor();

    Tensor(std::string name);

    explicit Tensor(std::vector<int> shape);

    ~Tensor();

    Tensor(const Tensor &t) = default;

    Tensor &operator=(DependencyAdder dep_adder);

    Tensor &operator=(InitDependencyAdder dep_adder);

    void Initialize(Tensor init);

    std::vector<OpDependency> OpDependencies() const;

    OpDependency InitDependency() const;

private:
    struct InternalData;
    std::shared_ptr<InternalData> internal_data;
};

class Tensor::DependencyAdder{
public:
    DependencyAdder(OpDependency dep);

protected:
    void ExtractDepFromInputs();

    std::vector<OpDependency> RemoveDuplicatedDep(std::vector<OpDependency> &deps);

    bool IsIn(const OpDependency &target);

private:
    friend class Tensor;
    const OpDependency *target_dep;
    std::vector<OpDependency> all_deps;
};

class Tensor::InitDependencyAdder{
public:
    InitDependencyAdder(OpDependency dep);

private:
    friend class Tensor;
    const OpDependency *target_dep;
};

} // end namespace mlfe
#endif // end ifndef __TENSOR_HPP__
