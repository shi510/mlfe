#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../utils/assert.hpp"
#include "../../math/transform.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CUDAContext>
struct ConvCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        DataType dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _padding = oc->attr->GetParam<std::vector<int>>("Padding");
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _b = oc->inputs[2];
        _y = oc->outputs[0];
        _batch = _x->Dim(0);
        // Output Filters.
        _m = _w->Dim(0);
        // Output Feature Map Size.
        _n = _y->Dim(2) * _y->Dim(3);
        // Weight Size.
        _k = _w->Dim(1) * _w->Dim(2) * _w->Dim(3);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _w->Allocate(Accelerator::CUDA, dt);
        _b->Allocate(Accelerator::CUDA, dt);
        _y->Allocate(Accelerator::CUDA, dt);

        _bias_multiplier.Reshape({ _n });
        _bias_multiplier.Allocate(Accelerator::CUDA, dt);

        _col_buffer.Reshape({ _k, _n });
        _col_buffer.Allocate(Accelerator::CUDA, dt);

        math::set<T, D>(
            _bias_multiplier.Size(),
            static_cast<T>(1),
            _bias_multiplier.GetPtr<T>()
            );
    }

    void Run() override {
        const T *x_ptr = _x->GetPtr<T>();
        const T *w_ptr = _w->GetPtr<T>();
        T *y_ptr = _y->GetPtr<T>();
        T *col_ptr = _col_buffer.GetPtr<T>();

        for (int i = 0; i < _batch; ++i) {
            /*
            * image to column in range on kernel size.
            */
            math::im2col<T, D>(
                _x->Dim(1), _x->Dim(2), _x->Dim(3),
                _kernel[0], _kernel[1],
                _stride[0], _padding[0],
                x_ptr, col_ptr);

            /*
            * convolution with kernel.
            * kernel is learnable variable.
            * _w({filters, _kernel_size}) * x_col({_kernel_size, out_size})
            *  = _y({filters, out_size})
            */
            math::gemm<T, D>(false, false,
                _m, _n, _k,
                static_cast<T>(1), w_ptr, _k,
                col_ptr, _n,
                static_cast<T>(0), y_ptr, _n, nullptr);

            /*
            * add bias.
            * bias is learnable varaible.
            * _y = _y + _b
            */
            math::gemm<T, D>(
                false, false,
                _m, _n, 1,
                static_cast<T>(1), _b->GetPtr<T>(), 1
                , _bias_multiplier.GetPtr<T>(), _n,
                static_cast<T>(1), y_ptr, _n, nullptr
                );

            /*
            * next batch.
            */
            x_ptr += _x->Dim(1) * _x->Dim(2) * _x->Dim(3);
            y_ptr += _n * _m;
        }
    }

    Tensor _col_buffer, _bias_multiplier;
    Tensor *_x, *_w, *_b;
    Tensor *_y;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    std::vector<int> _padding;
    int _filters;
    int _m, _n, _k;
    int _batch;
};

REGIST_NODE_FUNCTOR(Conv, DataType::F32, Accelerator::CUDA, ConvCudaF<float>)
//REGIST_NODE_FUNCTOR(Conv, DataType::F64, Accelerator::CUDA, ConvCudaF<double>)

template <typename T, typename D = CUDAContext>
struct ConvGradCudaF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _padding = oc->attr->GetParam<std::vector<int>>("Padding");
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _dy = oc->inputs[2];
        _dw = oc->outputs[0];
        _db = oc->outputs[1];
        _dx = oc->outputs[2];
        _batch = _x->Dim(0);

        // Output Filters.
        _m = _w->Dim(0);
        // Output Feature Map Size.
        _n = _dy->Dim(2) * _dy->Dim(3);
        // Weight Size.
        _k = _w->Dim(1) * _w->Dim(2) * _w->Dim(3);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dw->Allocate(Accelerator::CUDA, dt);
        _db->Allocate(Accelerator::CUDA, dt);
        _dx->Allocate(Accelerator::CUDA, dt);

        _bias_multiplier.Reshape({ _n });
        _bias_multiplier.Allocate(Accelerator::CUDA, dt);
        _col_buffer.Reshape({ _k, _n });
        _col_buffer.Allocate(Accelerator::CUDA, dt);
        math::set<T, D>(
            _bias_multiplier.Size(),
            static_cast<T>(1),
            _bias_multiplier.GetPtr<T>()
            );
    }

    void Run() override {
        const T *x_ptr = _x->GetPtr<T>();
        const T *dy_ptr = _dy->GetPtr<T>();
        T *col_ptr = _col_buffer.GetPtr<T>();
        T *dx_ptr = _dx->GetPtr<T>();

        math::scal<T, D>(
            _dx->Size(), 
            static_cast<T>(0),
            _dx->GetPtr<T>(),
            _dx->GetPtr<T>()
            );
        math::scal<T, D>(
            _dw->Size(), 
            static_cast<T>(0),
            _dw->GetPtr<T>(),
            _dw->GetPtr<T>()
            );
        math::scal<T, D>(
            _db->Size(), 
            static_cast<T>(0),
            _db->GetPtr<T>(),
            _db->GetPtr<T>()
            );

        for (int i = 0; i < _batch; ++i) {
            /*
            * gradient w.r.t. bias.
            */
            math::gemv<T, D>(
                false, _m, _n,
                static_cast<T>(1), dy_ptr, _n,
                _bias_multiplier.GetPtr<T>(), static_cast<T>(1),
                _db->GetPtr<T>(), static_cast<T>(1), nullptr
                );

            math::im2col<T, D>(
                _x->Dim(1), _x->Dim(2), _x->Dim(3),
                _kernel[0], _kernel[1],
                _stride[0], _padding[0],
                x_ptr, col_ptr
                );

            /*
            * Calculate gradients of weights.
            * kernel_size = {kernel_h, kernel_w, channel_of_x} = k
            * filters = {number of feature map channel} = m
            * out_size = {y_h, y_w} = n
            * dy({filters, out_size}) * col({kernel_size, out_size})^T
            *  = dw({filters, kernel_size})
            */
            math::gemm<T, D>(
                false, true, _m, _k, _n,
                static_cast<T>(1), dy_ptr, _n,
                col_ptr, _n,
                static_cast<T>(1), _dw->GetPtr<T>(), _k, nullptr
                );

            /*
            * Calculate loss to propagate through bottom.
            * w({filters, kernel_size})^T * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, D>(
                true, false, _k, _n, _m,
                static_cast<T>(1), _w->GetPtr<T>(), _k,
                dy_ptr, _n,
                static_cast<T>(0), col_ptr, _n, nullptr
                );

            math::col2im<T, D>(
                col_ptr,
                _x->Dim(1), _x->Dim(2), _x->Dim(3),
                _kernel[0], _stride[0], _padding[0],
                dx_ptr
                );

            /*
            * next batch.
            */
            x_ptr += _x->Size() / _x->Dim(0);
            dx_ptr += _dx->Size() / _dx->Dim(0);
            dy_ptr += _n * _m;
        }

        //math::scal<T, D>(
        //    _db->Size(),
        //    T(1) / static_cast<T>(_batch),
        //    _db->GetPtr<T>(),
        //    _db->GetPtr<T>()
        //    );

        //math::scal<T, D>(
        //    _dw->Size(),
        //    T(1) / static_cast<T>(_batch),
        //    _dw->GetPtr<T>(),
        //    _dw->GetPtr<T>()
        //    );
    }

    Tensor _col_buffer, _bias_multiplier;
    Tensor *_x, *_w, *_dy;
    Tensor *_dw, *_db, *_dx;
    int _m, _n, _k;
    int _batch;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    std::vector<int> _padding;
};

REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F32, Accelerator::CUDA, ConvGradCudaF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F64, Accelerator::CUDA, ConvGradCudaF<double>)

} // end namespace node
} // end namespace mlfe