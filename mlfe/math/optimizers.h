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

template <class T, class Dev>
void adadelta(const int size,
              T *w,
              T *dw,
              T *grad_hist,
              T *acc_hist,
              T lr,
              T momentum,
              T eps
             );

template <class T, class Dev>
void adam(const int size,
          T *w,
          T *dw,
          T *m_hist,
          T *v_hist,
          T lr,
          T beta1,
          T beta2,
          T eps
        );

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_OPTIMIZAERS_H__
