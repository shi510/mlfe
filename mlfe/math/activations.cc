#include "activations.h"
#include "mlfe/device_context/cpu_context.h"
#include <cmath>
#include <algorithm>

namespace mlfe { namespace math {

template <>
void relu<float, CPUContext>(
                            const int size,
                            const float *x,
                            float *y
                            ){
    for (int i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

template <>
void relu<double, CPUContext>(const int size,
                              const double *x,
                              double *y
                             )
{
    for (int i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

template <>
void relu_gradient<float, CPUContext>(const int size,
                                      const float *y,
                                      const float *dy,
                                      float *dx
                                     )
{
    for (int i = 0; i < size; ++i) {
        dx[i] = y[i] > 0 ? dy[i] : 0;
    }
}

template <>
void relu_gradient<double, CPUContext>(const int size,
                                       const double *y,
                                       const double *dy,
                                       double *dx
                                      )
{
    for (int i = 0; i < size; ++i) {
        dx[i] = y[i] > 0 ? dy[i] : 0;
    }
}

template <>
void sigmoid<float, CPUContext>(const int size,
                                const float *x,
                                float *y
                               )
{
    for (int i = 0; i < size; ++i) {
        y[i] = 1.f / (1.f + std::exp(-x[i]));
    }
}

template <>
void sigmoid<double, CPUContext>(const int size,
                                 const double *x,
                                 double *y
                                )
{
    for (int i = 0; i < size; ++i) {
        y[i] = 1.f / (1.f + std::exp(-x[i]));
    }
}

template <>
void sigmoid_gradient<float, CPUContext>(const int size,
                                         const float *y,
                                         const float *dy,
                                         float *dx
                                        )
{
    for (int i = 0; i < size; ++i) {
        dx[i] = dy[i] * y[i] * (1.f - y[i]);
    }
}

template <>
void sigmoid_gradient<double, CPUContext>(const int size,
                                          const double *y,
                                          const double *dy,
                                          double *dx
                                         )
{
    for (int i = 0; i < size; ++i) {
        dx[i] = dy[i] * y[i] * (1. - y[i]);
    }
}

template <>
void cross_entropy<float, CPUContext>(const int m,
                                      const int n,
                                      const float *prob_ptr,
                                      const float *label_ptr,
                                      float *loss_ptr
                                     )
{
    using namespace std;
    for(int i = 0; i < m; ++i){
        float row_loss = 0.f;
        for(int j = 0; j < n; ++j){
            int idx = i * n + j;
            row_loss += log(max(prob_ptr[idx], 1e-20f)) * label_ptr[idx];
        }
        loss_ptr[i] = -row_loss;
    }
}

template <>
void cross_entropy<double, CPUContext>(const int m,
                                       const int n,
                                       const double *prob_ptr,
                                       const double *label_ptr,
                                       double *loss_ptr
                                      )
{
    using namespace std;
    for(int i = 0; i < m; ++i){
        double row_loss = 0.;
        for(int j = 0; j < n; ++j){
            int idx = i * n + j;
            row_loss += log(max(prob_ptr[idx], 1e-20)) * label_ptr[idx];
        }
        loss_ptr[i] = -row_loss;
    }
}

template <>
void cross_entropy_gradient<float, CPUContext>(const int m,
                                               const int n,
                                               const float *prob_ptr,
                                               const float *label_ptr,
                                               const float *loss_ptr,
                                               float *dx_ptr
                                              )
{
    for(int i = 0; i < m; ++i){
        float loss = loss_ptr[i];
        for(int j = 0; j < n; ++j) {
            int idx = i * n + j;
            dx_ptr[idx] = (prob_ptr[idx] - label_ptr[idx]) * loss;
        }
    }
}

template <>
void cross_entropy_gradient<double, CPUContext>(const int m, const int n,
                                                const double *prob_ptr,
                                                const double *label_ptr,
                                                const double *loss_ptr,
                                                double *dx_ptr
                                                )
{
    for(int i = 0; i < m; ++i){
        double loss = loss_ptr[i];
        for(int j = 0; j < n; ++j) {
            int idx = i * n + j;
            dx_ptr[idx] = (prob_ptr[idx] - label_ptr[idx]) * loss;
        }
    }
}

} /* namespace math */
} /* namespace mlfe */
