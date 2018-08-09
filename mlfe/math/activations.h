#ifndef __MATH_ACTIVATIONS_H__
#define __MATH_ACTIVATIONS_H__

namespace mlfe { namespace math {

template <class T, class Dev>
void relu(const int size, const T *x, T *y);

template <class T, class Dev>
void relu_gradient(const int size, const T *x, const T *dy, T *dx);

template <class T, class Dev>
void sigmoid(const int size, const T *x, T *y);

template <class T, class Dev>
void sigmoid_gradient(const int size, const T *y, const T *dy, T *dx);

template <class T, class Dev>
void cross_entropy(const int m,
                   const int n,
                   const T *prob_ptr,
                   const T *label_ptr,
                   T *loss_ptr
                  );

template <class T, class Dev>
void cross_entropy_gradient(const int m,
                            const int n,
                            const T *prob_ptr,
                            const T *label_ptr,
                            const T *loss_ptr,
                            T *dx_ptr
                           );

template <class T, class Dev>
void sigmoid_cross_entropy(const int m,
                           const int n,
                           const T *x_ptr,
                           const T *t_ptr,
                           T *loss_ptr
                          );

template <class T, class Dev>
void sigmoid_cross_entropy_gradient(const int m,
                                    const int n,
                                    const T *x_ptr,
                                    const T *t_ptr,
                                    const T *dy_ptr,
                                    T *dx_ptr
                                   );

template<class T, class Dev>
void reduce_mean_gradient(const int size,
                          const float scale,
                          const float *dy_ptr,
                          float *dx_ptr
                         );

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_ACTIVATIONS_H__
