#ifndef __ADAM_H__
#define __ADAM_H__
#include "optimizer.h"

namespace mlfe{
namespace functional{

opt::optimizer_ptr create_adam_optimizer(double lr,
                                         double beta1 = 0.9,
                                         double beta2 = 0.999,
                                         double eps = 1e-8
                                         );

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __ADAM_H__
