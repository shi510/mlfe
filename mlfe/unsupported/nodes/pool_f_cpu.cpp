#include <algorithm>
#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct MaxPoolCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _x = oc->inputs[0];
        _y = oc->outputs[0];
        _idx = oc->outputs[1];
        _batch = _x->Dim(0);
        _in_c = _x->Dim(1);
        _out_h = _y->Dim(2);
        _out_w = _y->Dim(3);
        _y->Allocate();
        _idx->Allocate();
    }

    void Run() override {
        const T *x_ptr = _x->GetPtr<T>();
        T *y_ptr = _y->GetPtr<T>();
        int *idx_ptr = _idx->GetPtr<int>();

        math::set<T, D>(
            _y->Size(),
            static_cast<T>(-FLT_MAX),
            _y->GetPtr<T>()
            );

        for (int n = 0; n < _batch; ++n) {
            for (int c = 0; c < _in_c; ++c) {
                for (int ph = 0; ph < _out_h; ++ph) {
                    for (int pw = 0; pw < _out_w; ++pw) {
                        int hstart = ph * _stride[0];
                        int wstart = pw * _stride[1];
                        int hend = std::min<int>(hstart + _kernel[0], _x->Dim(2));
                        int wend = std::min<int>(wstart + _kernel[1], _x->Dim(3));
                        const int pool_index = ph * _out_w + pw;
                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                const int index = h * _x->Dim(3) + w;
                                if (x_ptr[index] > y_ptr[pool_index]) {
                                    y_ptr[pool_index] = x_ptr[index];
                                    idx_ptr[pool_index] = index;
                                }
                            }
                        }
                    }
                }
                x_ptr += _x->Dim(2) * _x->Dim(3);
                y_ptr += _out_h * _out_w;
                idx_ptr += _out_h * _out_w;
            }
        }
    }

    Tensor *_x;
    Tensor *_idx, *_y;
    int _batch;
    int _in_c;
    int _in_h, _in_w;
    int _out_h, _out_w;
    std::vector<int> _kernel;
    std::vector<int> _stride;
};

REGIST_NODE_FUNCTOR(MaxPool, DataType::F32, Accelerator::Default, MaxPoolCpuF<float>)
REGIST_NODE_FUNCTOR(MaxPool, DataType::F64, Accelerator::Default, MaxPoolCpuF<double>)

template <typename T, typename D = CPUContext>
struct MaxPoolGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _idx = oc->inputs[2];
        _dy = oc->inputs[3];
        _dx = oc->outputs[0];
        _batch = _dy->Dim(0);
        _in_c = _dx->Dim(1);
        _out_h = _idx->Dim(2);
        _out_w = _idx->Dim(3);
        _dx->Allocate();
    }

    void Run() override {
        const T *dy_ptr = _dy->GetPtr<T>();
        const int *idx_ptr = _idx->GetPtr<int>();
        T *dx_ptr = _dx->GetPtr<T>();

        math::set<T, D>(
            _dx->Size(),
            static_cast<T>(0),
            _dx->GetPtr<T>()
            );

        for (int n = 0; n < _batch; ++n) {
            for (int c = 0; c < _in_c; ++c) {
                for (int ph = 0; ph < _out_h; ++ph) {
                    for (int pw = 0; pw < _out_w; ++pw) {
                        const int index = ph * _out_w + pw;
                        const int bottom_index = idx_ptr[index];
                        dx_ptr[bottom_index] += dy_ptr[index];
                    }
                }
                dx_ptr += _dx->Dim(2) * _dx->Dim(3);
                dy_ptr += _dy->Dim(2) * _dy->Dim(3);
                idx_ptr += _idx->Dim(2) * _idx->Dim(3);
            }
        }
    }
    Tensor *_idx, *_dy;
    Tensor *_dx;
    int _batch;
    int _in_c;
    int _in_h, _in_w;
    int _out_h, _out_w;
};

REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F32, Accelerator::Default, MaxPoolGradCpuF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(MaxPool, DataType::F64, Accelerator::Default, MaxPoolGradCpuF<double>)

} // end namespace node
} // end namespace mlfe