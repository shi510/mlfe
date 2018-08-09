#include "blas.h"
#include "../device_context/cpu_context.h"
#include <Eigen/Dense>

namespace mlfe{ namespace math{

template<>
void gemm<float, CPUContext>(const bool trans_a,
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
                            )
{
    using ConstMapMatXf = Eigen::Map<const Eigen::MatrixXf>;
    Eigen::Map<Eigen::MatrixXf> c(c_ptr, n, m);
    if(beta == 0.f){
        c.setZero();
    }
    else{
        c *= beta;
    }
    if(!trans_a && !trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, n, k) *
                                ConstMapMatXf(a_ptr, k, m)
                               );
    }
    else if(trans_a && !trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, n, k) *
                                ConstMapMatXf(a_ptr, m, k).transpose()
                               );
    }
    else if(!trans_a && trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, k, n).transpose() *
                                ConstMapMatXf(a_ptr, k, m)
                               );
    }
    else{
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, k, n).transpose() *
                                ConstMapMatXf(a_ptr, m, k).transpose()
                               );
    }
}

template<>
void gemm<double, CPUContext>(const bool trans_a,
                              const bool trans_b,
                              const int m,
                              const int n,
                              const int k,
                              const double alpha,
                              const double *a_ptr,
                              const int lda,
                              const double *b_ptr,
                              const int ldb,
                              const double beta,
                              double *c_ptr,
                              const int ldc,
                              CPUContext *context
                             )
{
    using ConstMapMatXf = Eigen::Map<const Eigen::MatrixXd>;
    Eigen::Map<Eigen::MatrixXd> c(c_ptr, n, m);
    if(beta == 0.){
        c.setZero();
    }
    else{
        c *= beta;
    }
    if(!trans_a && !trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, n, k) *
                                ConstMapMatXf(a_ptr, k, m)
                               );
    }
    else if(trans_a && !trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, n, k) *
                                ConstMapMatXf(a_ptr, m, k).transpose()
                               );
    }
    else if(!trans_a && trans_b){
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, k, n).transpose() *
                                ConstMapMatXf(a_ptr, k, m)
                               );
    }
    else{
        c.noalias() += alpha * (ConstMapMatXf(b_ptr, k, n).transpose() *
                                ConstMapMatXf(a_ptr, m, k).transpose()
                               );
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
                            )
{
    using ConstMapMatXf = Eigen::Map<const Eigen::MatrixXf>;
    using ConstMapVecXf = Eigen::Map<const Eigen::VectorXf>;
    Eigen::Map<Eigen::VectorXf> c(c_ptr, !trans_a ? m : n);
    if(beta == 0.f){
        c.setZero();
    }
    else{
        c *= beta;
    }

    if(!trans_a){
        c.noalias() += alpha * (ConstMapMatXf(a_ptr, n, m).transpose() *
                                ConstMapVecXf(b_ptr, n)
                                );
    }
    else{
        c.noalias() += alpha * (ConstMapMatXf(a_ptr, n, m) *
                                ConstMapVecXf(b_ptr, m)
                                );
    }
}

template <>
void gemv<double, CPUContext>(const bool trans_a,
                              const int m,
                              const int n,
                              const double alpha,
                              const double *a_ptr,
                              const int lda,
                              const double *b_ptr,
                              const double beta,
                              double *c_ptr,
                              const int ldc,
                              CPUContext *context
                             )
{
    using ConstMapMatXf = Eigen::Map<const Eigen::MatrixXd>;
    using ConstMapVecXf = Eigen::Map<const Eigen::VectorXd>;
    Eigen::Map<Eigen::VectorXd> c(c_ptr, !trans_a ? m : n);
    if(beta == 0.){
        c.setZero();
    }
    else{
        c *= beta;
    }

    if(!trans_a){
        c.noalias() += alpha * (
                                ConstMapMatXf(a_ptr, n, m).transpose() *
                                ConstMapVecXf(b_ptr, n)
                                );
    }
    else{
        c.noalias() += alpha * (
                                ConstMapMatXf(a_ptr, n, m) *
                                ConstMapVecXf(b_ptr, m)
                                );
    }
}

} // end namespace math
} // end namespace mlfe
