#ifndef __ADADELTA_H__
#define __ADADELTA_H__
#include "optimizer.h"

namespace mlfe{ namespace optimizer{

class AdaDelta : public Optimizer{
public:
    AdaDelta(double learning_rate, double momentum_rate, double epsilon = 1e-8);

    AppliedOptimizer Minimize(Tensor loss) override;

private:
    double lr;
    double mr;
    double eps;
};

} // end namespace optimizer
} // end namespace mlfe
#endif // end #ifndef __ADADELTA_H__
