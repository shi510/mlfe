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

template <>
void rowwise_max<float, CPUContext>(
                                    const int m, const int n,
                                    const float *a_ptr,
                                    float *b_ptr
                                    ){
    Eigen::Map<Eigen::VectorXf>(b_ptr, n) =  Eigen::Map<const Eigen::MatrixXf>(a_ptr, n, m).colwise().maxCoeff();
}

template <>
void rowwise_normalize<float, CPUContext>(
                                          const int m, const int n,
                                          const float *scaler_ptr,
                                          float *norm_dest
                                          ){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            norm_dest[i * n + j] /= scaler_ptr[i];
        }
    }
}

template <>
void cross_entropy<float, CPUContext>(
                                      const int m, const int n,
                                      const float *prob_ptr,
                                      const float *label_ptr,
                                      float *loss_ptr
                                      ){
    loss_ptr[0] = 0.f;
    for(int i = 0; i < m; ++i){
        float row_loss = 0.f;
        for(int j = 0; j < n; ++j){
            row_loss += -std::log(std::max(prob_ptr[i * n + j], 1e-20f)) * label_ptr[i * n + j];
        }
        loss_ptr[0] += row_loss;
    }
    loss_ptr[0] /= m;
}

template <>
void cross_entropy_gradients<float, CPUContext>(
                                                const int m, const int n,
                                                const float *prob_ptr,
                                                const float *label_ptr,
                                                float *dx_ptr
                                                ){
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            dx_ptr[idx] = prob_ptr[idx] - label_ptr[idx];
        }
    }
}

template<>
void exp<float, CPUContext>(
                            const int size,
                            const float *x_ptr,
                            float *y_ptr){
    Eigen::Map<Eigen::VectorXf>(y_ptr, size) = Eigen::Map<const Eigen::VectorXf>(x_ptr, size).array().exp();
}

template<>
void axpy<float, CPUContext>(int _size,
                             const float _alpha,
                             const float *_x,
                             float *_y){
    Eigen::Map<Eigen::VectorXf>(_y, _size) += _alpha * Eigen::Map<const Eigen::VectorXf>(_x, _size);
}

template <>
void scal<float, CPUContext>(const int size,
                              const float alpha,
                              const float *x_ptr,
                              float *y_ptr){
    Eigen::Map<Eigen::VectorXf> y(y_ptr, size);
    if(alpha != 0){
        y = alpha * Eigen::Map<const Eigen::VectorXf>(x_ptr, size);
    }
    else{
        y.setZero();
    }
}

} /* math */
} /* mlfe */
