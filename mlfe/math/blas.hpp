#ifndef __MATH_BLAS3_HPP__
#define __MATH_BLAS3_HPP__

namespace mlfe{ namespace math{

template<class DataType, class DeviceContext>
void gemm(
          const bool trans_a, const bool trans_b,
          const int m, const int n, const int k,
          const DataType alpha,
          const DataType *a, const int lda,
          const DataType *b, const int ldb,
          const DataType beta,
          DataType *c, const int ldc,
          DeviceContext *context
          );

template<class DataType, class DeviceContext>
void gemv(
          const bool trans_a,
          const int m, const int n,
          const DataType alpha,
          const DataType *a, const int lda,
          const DataType *b,
          const DataType beta,
          DataType *c, const int ldc,
          DeviceContext *context
          );
    
template <class DataType, class DeviceContext>
void rowwise_max(
                 const int m, const int n,
                 const DataType *a_ptr,
                 DataType *b_ptr
                 );
    
template <class DataType, class DeviceContext>
void rowwise_normalize(
                       const int m, const int n,
                       const DataType *scaler_ptr,
                       DataType *norm_dest
                       );
    
template <class DataType, class DeviceContext>
void cross_entropy(
                   const int m, const int n,
                   const DataType *prob_ptr,
                   const DataType *label_ptr,
                   DataType *loss_ptr
                   );
    
template <class DataType, class DeviceContext>
void cross_entropy_gradients(
                             const int m, const int n,
                             const DataType *prob_ptr,
                             const DataType *label_ptr,
                             DataType *dx_ptr
                             );
    
template<class DataType, class DeviceContext>
void exp(
         const int size,
         const DataType *x_ptr,
         DataType *y_ptr
         );
    
template<class DataType, class DeviceContext>
void axpy(
          int size,
          DataType alpha,
          const DataType *x_ptr,
          DataType *y_ptr);
    
template<class DataType, class DeviceContext>
void scal(
          const int size,
          const DataType alpha,
          const DataType *x_ptr,
          DataType *y_ptr);

} /* namespace math */
} /* namespace mlfe */
#endif /* __MATH_BLAS3_HPP__ */
