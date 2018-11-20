#ifndef __ADADELTA_H__
#define __ADADELTA_H__
#include "optimizer.h"

namespace mlfe{
namespace functional{

opt::optimizer_ptr create_adadelta_optimizer(double lr,
                                             double momentum,
                                             double eps = 1e-8
                                             );

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __ADADELTA_H__
