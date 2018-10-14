#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "../core/tensor.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"
#include <vector>

namespace mlfe{ namespace optimizer{
class AppliedOptimizer;

class Optimizer{
using TensorPairs = std::vector<std::pair<Tensor, Tensor>>;
using OpDeps = std::vector<OpDependency>;
public:
    class UpdateRule;

    virtual AppliedOptimizer Minimize(Tensor loss) = 0;

    TensorPairs compute_gradient(const Tensor root);

    OpDeps get_op_deps() const;

protected:

    AppliedOptimizer ApplyGradientUpdateRule(UpdateRule ur);

    OpDeps bwd_op_deps;
};

class Optimizer::UpdateRule{
public:
    UpdateRule(Tensor target, OpDeps bwd_deps);

    UpdateRule(Tensor target, Tensor dx);

    void AddRule(OpDependency od);

private:
    friend class AppliedOptimizer;
    Tensor opt_target;
    std::vector<OpDependency> _bwd_op_deps;
    std::vector<OpDependency> _update_op_deps;
};

class AppliedOptimizer{
public:
    Tensor Target() const;

    std::vector<OpDependency> get_bwd_op_deps() const;

    std::vector<OpDependency> get_update_op_deps() const;

protected:
    AppliedOptimizer() = default;

    AppliedOptimizer(Optimizer::UpdateRule ur);

private:
    friend class Optimizer;
    Optimizer::UpdateRule rule;
};

} // end namespace optimizer
} // end namespace mlfe
#endif // end ifndef __OPTIMIZER_HPP__
