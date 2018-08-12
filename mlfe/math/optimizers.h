#ifndef __MATH_OPTIMIZAERS_H__
#define __MATH_OPTIMIZAERS_H__

namespace mlfe { namespace math {

template <class T, class Dev>
void gradient_descent_momentum(const int size,
                               T *w,
                               T *dw,
                               T *w_momentum,
                               T lr,
                               T momentum,
                               T decay
                              );

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_OPTIMIZAERS_H__
