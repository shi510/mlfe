#ifndef __MATH_BASIC_FUNCTIONS_H__
#define __MATH_BASIC_FUNCTIONS_H__

namespace mlfe{ namespace math{

template <class T, class Dev>
void rowwise_max(const int m,
                 const int n,
                 const T *a_ptr,
                 T *b_ptr
                );

template <class T, class Dev>
void rowwise_normalize(const int m,
                       const int n,
                       const T *scaler_ptr,
                       T *norm_dest
                      );

template <class T, class Dev>
void exp(const int size,
         const T *x_ptr,
         T *y_ptr
        );

template <class T, class Dev>
void axpy(int size,
          const T alpha,
          const T *x_ptr,
          T *y_ptr
         );

template <class T, class Dev>
void scal(const int size,
          const T alpha,
          const T *x_ptr,
          T *y_ptr
         );

template <class T, class Dev>
void sum(const int size,
         const T *x_ptr,
         T *y_ptr
        );

template <class T, class Dev>
void set(const int size,
         const T val,
         T *x_ptr
        );

template<class T, class Dev>
void elementwise_mul(const int size,
                     const T *a,
                     const T *b,
                     T *c
                    );

template <class T, class Dev>
void clip_min_max(const int size,
                  T *data,
                  T min,
                  T max
                  );

template <class T, class Dev>
void shift_a_b(const int size,
               T *x,
               T a,
               T b
              );

template <typename T>
void OneHotCuda(const int batch,
                const int classes,
                const T *label,
                T *onehot
               );

template <typename T>
void AccuracyCuda(const int batch,
                  const int classes,
                  const int top_k,
                  const T *prob,
                  const T *label,
                  T *accuracy
                 );

template <class T, class Dev>
void bernoulli_distribution(const int size,
                            const T prob,
                            T *bernoulli
                           );

#define DECLARE_CUDA_BINARY_OP(OpName)    \
template <typename T>                     \
void OpName##Cuda(const int size,         \
                  const T *a,             \
                  const T *b,             \
                  T *c                    \
                 );                       \
template <typename T>                     \
void OpName##ValCuda(const int size,      \
                     const T val,         \
                     const T *a,          \
                     T *c                 \
                    );

DECLARE_CUDA_BINARY_OP(Add)
DECLARE_CUDA_BINARY_OP(Sub)
DECLARE_CUDA_BINARY_OP(Mul)
DECLARE_CUDA_BINARY_OP(Div)

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_BASIC_FUNCTIONS_H__
