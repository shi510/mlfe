#ifndef __GRADIENT_DESCENT_H__
#define __GRADIENT_DESCENT_H__
#include "optimizer.h"

namespace mlfe{
namespace functional{

opt::optimizer_ptr create_gradient_descent(double lr, double momentum);

} // end namespace functional
} // end namespace mlfe
#endif // end #ifndef __GRADIENT_DESCENT_HPP__
