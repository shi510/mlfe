#include "basic_functions.h"
#include "mlfe/device_context/cpu_context.h"
#include <Eigen/Dense>

namespace mlfe{ namespace math{

template <>
void squared_difference<float, CPUContext>(const int size,
                                           const float *x1_ptr,
                                           const float *x2_ptr,
                                           float *y_ptr){
    for(int n = 0; n < size; ++n){
        y_ptr[n] = std::pow(x1_ptr[n] - x2_ptr[n], 2);
    }
}

template <>
void squared_difference<double, CPUContext>(const int size,
                                            const double *x1_ptr,
                                            const double *x2_ptr,
                                            double *y_ptr){
    for(int n = 0; n < size; ++n){
        y_ptr[n] = std::pow(x1_ptr[n] - x2_ptr[n], 2);
    }
}

template <>
void rowwise_max<float, CPUContext>(const int m,
                                    const int n,
                                    const float *a_ptr,
                                    float *b_ptr
                                    )
{
    Eigen::Map<Eigen::VectorXf>(b_ptr, m) =
    Eigen::Map<const Eigen::MatrixXf>(a_ptr, n, m).colwise().maxCoeff();
}

template <>
void rowwise_max<double, CPUContext>(const int m,
                                     const int n,
                                     const double *a_ptr,
                                     double *b_ptr
                                     )
{
    Eigen::Map<Eigen::VectorXd>(b_ptr, m) =
    Eigen::Map<const Eigen::MatrixXd>(a_ptr, n, m).colwise().maxCoeff();
}

template <>
void rowwise_normalize<float, CPUContext>(const int m, const int n,
                                          const float *scaler_ptr,
                                          float *norm_dest
                                          )
{
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            norm_dest[i * n + j] /= scaler_ptr[i];
        }
    }
}

template <>
void rowwise_normalize<double, CPUContext>(const int m,
                                           const int n,
                                           const double *scaler_ptr,
                                           double *norm_dest
                                           )
{
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            norm_dest[i * n + j] /= scaler_ptr[i];
        }
    }
}


template<>
void exp<float, CPUContext>(const int size,
                            const float *x_ptr,
                            float *y_ptr
                           )
{
    Eigen::Map<Eigen::VectorXf>(y_ptr, size) =
    Eigen::Map<const Eigen::VectorXf>(x_ptr, size).array().exp();
}

template<>
void exp<double, CPUContext>(const int size,
                             const double *x_ptr,
                             double *y_ptr
                            )
{
    Eigen::Map<Eigen::VectorXd>(y_ptr, size) =
    Eigen::Map<const Eigen::VectorXd>(x_ptr, size).array().exp();
}

template<>
void axpy<float, CPUContext>(int size,
                             const float alpha,
                             const float *x_ptr,
                             float *y_ptr
                            )
{
    Eigen::Map<Eigen::VectorXf>(y_ptr, size) +=
    alpha * Eigen::Map<const Eigen::VectorXf>(x_ptr, size);
}

template<>
void axpy<double, CPUContext>(int size,
                              const double alpha,
                              const double *x_ptr,
                              double *y_ptr
                             )
{
    Eigen::Map<Eigen::VectorXd>(y_ptr, size) +=
    alpha * Eigen::Map<const Eigen::VectorXd>(x_ptr, size);
}

template <>
void scal<float, CPUContext>(const int size,
                             const float alpha,
                             const float *x_ptr,
                             float *y_ptr
                            )
{
    Eigen::Map<Eigen::VectorXf> y(y_ptr, size);
    if(alpha != 0.f){
        y = alpha * Eigen::Map<const Eigen::VectorXf>(x_ptr, size);
    }
    else{
        y.setZero();
    }
}

template <>
void scal<double, CPUContext>(const int size,
                              const double alpha,
                              const double *x_ptr,
                              double *y_ptr
                             )
{
    Eigen::Map<Eigen::VectorXd> y(y_ptr, size);
    if(alpha != 0.){
        y = alpha * Eigen::Map<const Eigen::VectorXd>(x_ptr, size);
    }
    else{
        y.setZero();
    }
}

template <>
void sum<float, CPUContext>(
                            const int size,
                            const float *x_ptr,
                            float *y_ptr
                           )
{
    y_ptr[0] = Eigen::Map<const Eigen::VectorXf>(x_ptr, size).sum();
}

template <>
void sum<double, CPUContext>(
                             const int size,
                             const double *x_ptr,
                             double *y_ptr
                            )
{
    y_ptr[0] = Eigen::Map<const Eigen::VectorXd>(x_ptr, size).sum();
}

template<>
void set<float, CPUContext>(
                            const int size,
                            const float val,
                            float *x_ptr
                           )
{
    Eigen::Map<Eigen::VectorXf>(x_ptr, size).setConstant(val);
}

template<>
void set<double, CPUContext>(const int size,
                             const double val,
                             double *x_ptr
                             )
{
    Eigen::Map<Eigen::VectorXd>(x_ptr, size).setConstant(val);
}

} // end namespace math
} // end namespace mlfe
