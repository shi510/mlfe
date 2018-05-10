#include <unsupported/Eigen/CXX11/Tensor>
#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include "../../math/transform.hpp"
#include "../../utils/assert.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct ConvCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        DataType dt = DataType::F32;
        _kernel = oc->attr->GetParam<std::vector<int>>("Kernel");
        _stride = oc->attr->GetParam<std::vector<int>>("Stride");
        _padding = oc->attr->GetParam<std::vector<int>>("Padding");
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _b = oc->inputs[2];
        _y = oc->outputs[0];
        
        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _w->Allocate(Accelerator::Default, dt);
        _b->Allocate(Accelerator::Default, dt);
        _y->Allocate(Accelerator::Default, dt);
    }

    void Run() override {
        Eigen::Tensor<T, 4, Eigen::RowMajor> x_t = Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
            _x->GetPtr<T>(),
            _x->Dim(0),
            _x->Dim(1),
            _x->Dim(2),
            _x->Dim(3)
            ).shuffle(Eigen::array<int, 4>{{0, 2, 3, 1}});

        Eigen::Tensor<T, 4, Eigen::RowMajor> kernel_t = Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
            _w->GetPtr<T>(),
            _w->Dim(0),
            _w->Dim(1),
            _w->Dim(2),
            _w->Dim(3)
            ).shuffle(Eigen::array<int, 4>{ {2, 3, 1, 0}});

        Eigen::Tensor<T, 4, Eigen::RowMajor> y_t(_y->Dim(0), _y->Dim(2), _y->Dim(3), _y->Dim(1));

        y_t = x_t.extract_image_patches(
            _w->Dim(2),
            _w->Dim(3),
            _stride[0], _stride[1],
            1, 1,
            1, 1,
            _padding[0], _padding[0],
            _padding[1], _padding[1],
            0)
            .reshape(Eigen::array<int, 2>{ {_y->Size() / _y->Dim(1), _w->Size() / _w->Dim(0)}})
            .contract(
                kernel_t.reshape(Eigen::array<int, 2>{ {_w->Size() / _w->Dim(0), _w->Dim(0)}}),
                Eigen::array<Eigen::IndexPair<int>, 1>{ {Eigen::IndexPair<int>(1, 0)}}
        ).reshape(y_t.dimensions());

        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> y_arr(y_t.data(), _y->Dim(1), _y->Size() / _y->Dim(1));
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> b_arr(_b->GetPtr<T>(), _b->Size(), 1);
        y_arr = y_arr.colwise() + b_arr;

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
            _y->GetPtr<T>(),
            _y->Dim(0),
            _y->Dim(1),
            _y->Dim(2),
            _y->Dim(3)
            ) = y_t.shuffle(Eigen::array<int, 4>{{0, 3, 1, 2 }});
    }
    
    Tensor *_x, *_w, *_b;
    Tensor *_y;
    std::vector<int> _kernel;
    std::vector<int> _stride;
    std::vector<int> _padding;
};

REGIST_NODE_FUNCTOR(Conv, DataType::F32, Accelerator::Default, ConvCpuF<float>)
//REGIST_NODE_FUNCTOR(Conv, DataType::F64, Accelerator::Default, ConvCpuF<double>)

template <typename T, typename D = CPUContext>
struct ConvGradCpuF : NodeFunctor {
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
        _dw->Allocate(Accelerator::Default, dt);
        _db->Allocate(Accelerator::Default, dt);
        _dx->Allocate(Accelerator::Default, dt);

        _bias_multiplier.Reshape({ _n });
        _bias_multiplier.Allocate();
        _col_buffer.Reshape({ _k, _n });
        _col_buffer.Allocate();
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
            _dx->Size(), T(0),
            _dx->GetPtr<T>(),
            _dx->GetPtr<T>()
            );
        math::scal<T, D>(
            _dw->Size(), T(0),
            _dw->GetPtr<T>(),
            _dw->GetPtr<T>()
            );
        math::scal<T, D>(
            _db->Size(), T(0),
            _db->GetPtr<T>(),
            _db->GetPtr<T>()
            );

        for (int i = 0; i < _batch; ++i) {
            /*
            * gradient w.r.t. bias.
            */
            math::gemv<T, D>(
                false, _m, _n,
                T(1), dy_ptr, _n,
                _bias_multiplier.GetPtr<T>(), T(1),
                _db->GetPtr<T>(), T(1), nullptr
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
                T(1), dy_ptr, _n,
                col_ptr, _n,
                T(1), _dw->GetPtr<T>(), _k, nullptr
                );

            /*
            * Calculate loss to propagate through bottom.
            * w({filters, kernel_size})^T * dy({filters, out_size})
            *  = col({kernel_size, out_size})
            */
            math::gemm<T, D>(
                true, false, _k, _n, _m,
                T(1), _w->GetPtr<T>(), _k,
                dy_ptr, _n,
                T(0), col_ptr, _n, nullptr
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

REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F32, Accelerator::Default, ConvGradCpuF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(Conv, DataType::F64, Accelerator::Default, ConvGradCpuF<double>)

} // end namespace node
} // end namespace mlfe