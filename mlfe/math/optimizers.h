#ifndef __MATH_OPTIMIZAERS_H__
#define __MATH_OPTIMIZAERS_H__

namespace mlfe { namespace math {

template <class T, class Dev>
void gradient_descent_momentum(const int size,
                               T *w,
                               const T *dw,
                               T *w_momentum,
                               const T lr,
                               const T momentum,
                               const T decay
                              );

template <class T, class Dev>
void adadelta(const int size,
              T *w,
              const T *dw,
              T *grad_hist,
              T *acc_hist,
              const T lr,
              const T momentum,
              const T eps
             );

template <class T, class Dev>
void adam(const int size,
          T *w,
          const T *dw,
          T *m_hist,
          T *v_hist,
          const T lr,
          const T beta1,
          const T beta2,
          const T eps
        );

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_OPTIMIZAERS_H__
