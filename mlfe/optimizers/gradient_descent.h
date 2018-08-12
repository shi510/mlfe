#ifndef __GRADIENT_DESCENT_H__
#define __GRADIENT_DESCENT_H__
#include "optimizer.h"

namespace mlfe{ namespace optimizer{

class GradientDescent : public Optimizer{
public:
    GradientDescent(double learning_rate);

    AppliedOptimizer Minimize(Tensor loss) override;

private:
    double lr;
};

class GradientDescentWithMomentum : public Optimizer{
public:
    GradientDescentWithMomentum(double learning_rate,
                                double momentum_rate,
                                double weight_decay
                               );

    AppliedOptimizer Minimize(Tensor loss) override;

private:
    double lr;
    double mr;
    double wd;
};

} // end namespace optimizer
} // end namespace mlfe
#endif // end #ifndef __GRADIENT_DESCENT_HPP__
