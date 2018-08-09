#ifndef __MATH_BLAS3_HPP__
#define __MATH_BLAS3_HPP__

namespace mlfe{ namespace math{

template<class DataType, class DeviceContext>
void gemm(const bool trans_a, const bool trans_b,
          const int m, const int n, const int k,
          const DataType alpha,
          const DataType *a, const int lda,
          const DataType *b, const int ldb,
          const DataType beta,
          DataType *c, const int ldc,
          DeviceContext *context
          );

template<class DataType, class DeviceContext>
void gemv(const bool trans_a,
          const int m, const int n,
          const DataType alpha,
          const DataType *a, const int lda,
          const DataType *b,
          const DataType beta,
          DataType *c, const int ldc,
          DeviceContext *context
          );

} // end namespace math
} // end namespace mlfe
#endif // end #ifndef __MATH_BLAS3_HPP__
