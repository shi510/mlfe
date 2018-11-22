#ifndef __MATH_OP_H__
#define __MATH_OP_H__

namespace mlfe{
// forward declaration.
class Tensor;

namespace functional{

Tensor squared_difference(Tensor x1, Tensor x2);

Tensor mean(Tensor x);

} // end namespace functional
} // end namespace mlfe
#endif // end ifndef __MATH_OP_H__
