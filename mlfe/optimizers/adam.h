#ifndef __ADAM_H__
#define __ADAM_H__
#include "optimizer.h"

namespace mlfe{ namespace optimizer{

class Adam : public Optimizer{
public:
    Adam(double learning_rate,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-8
        );

    AppliedOptimizer Minimize(Tensor loss) override;

private:
    double lr;
    double b1;
    double b2;
    double eps;
};

} // end namespace optimizer
} // end namespace mlfe
#endif // end #ifndef __ADAM_H__
