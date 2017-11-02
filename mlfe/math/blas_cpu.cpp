#include <Eigen/Dense>
#include "blas.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe{ namespace math{

template<>
void gemm<float, CPUContext>(
                             const bool trans_a,
                             const bool trans_b,
                             const int m,
                             const int n,
                             const int k,
                             const float alpha,
                             const float *a_ptr,
                             const int lda,
                             const float *b_ptr,
                             const int ldb,
                             const float beta,
                             float *c_ptr,
                             const int ldc,
                             CPUContext *context
                             ){
    Eigen::Map<Eigen::MatrixXf> c(c_ptr, n, m);
    if(beta == 0.f){
        c.setZero();
    }
    else{
        c *= beta;
    }
    if(!trans_a && !trans_b){
        c.noalias() += alpha * (
                                 Eigen::Map<const Eigen::MatrixXf>(b_ptr, n, k) *
                                 Eigen::Map<const Eigen::MatrixXf>(a_ptr, k, m));
    }
    else if(trans_a && !trans_b){
        c.noalias() += alpha * (
                                 Eigen::Map<const Eigen::MatrixXf>(b_ptr, n, k) *
                                 Eigen::Map<const Eigen::MatrixXf>(a_ptr, m, k).transpose());
    }
    else if(!trans_a && trans_b){
        c.noalias() += alpha * (
                                 Eigen::Map<const Eigen::MatrixXf>(b_ptr, k, n).transpose() *
                                 Eigen::Map<const Eigen::MatrixXf>(a_ptr, k, m));
    }
    else{
        c.noalias() += alpha * (
                                 Eigen::Map<const Eigen::MatrixXf>(b_ptr, k, n).transpose() *
                                 Eigen::Map<const Eigen::MatrixXf>(a_ptr, m, k).transpose());
    }
}

template <>
void gemv<float, CPUContext>(const bool trans_a,
                             const int m,
                             const int n,
                             const float alpha,
                             const float *a_ptr,
                             const int lda,
                             const float *b_ptr,
                             const float beta,
                             float *c_ptr,
                             const int ldc,
                             CPUContext *context
                             ){
    Eigen::Map<Eigen::VectorXf> c(c_ptr, !trans_a ? m : n);
    if(beta == 0.f){
        c.setZero();
    }
    else{
        c *= beta;
    }
    
    if(!trans_a){
        c.noalias() += alpha * (
                                Eigen::Map<const Eigen::MatrixXf>(a_ptr, n, m).transpose() *
                                Eigen::Map<const Eigen::VectorXf>(b_ptr, n)
                                );
    }
    else{
        c.noalias() += alpha * (
                                Eigen::Map<const Eigen::MatrixXf>(a_ptr, n, m) *
                                Eigen::Map<const Eigen::VectorXf>(b_ptr, m)
                                );
    }
    
}

} /* math */
} /* mlfe */
