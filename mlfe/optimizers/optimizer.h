#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "../core/tensor.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"
#include <vector>

namespace mlfe{ namespace optimizer{
class AppliedOptimizer;

class Optimizer{
public:
    class UpdateRule;

    virtual AppliedOptimizer Minimize(Tensor loss) = 0;

protected:
    GradientHelper::HelperOut LetGradientFlowBack(Tensor loss);

    AppliedOptimizer ApplyGradientUpdateRule(UpdateRule ur);

private:
    friend class AppliedOptimizer;
};

class Optimizer::UpdateRule{
public:
    UpdateRule(Tensor target, Tensor dx);

    void AddRule(OpDependency od);

private:
    friend class AppliedOptimizer;
    Tensor opt_target;
    Tensor opt_dx;
    std::vector<OpDependency> deps;
};

class AppliedOptimizer{
public:
    Tensor Target() const;

    Tensor InputGrad() const;

    std::vector<OpDependency> OpDependencies() const;

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
