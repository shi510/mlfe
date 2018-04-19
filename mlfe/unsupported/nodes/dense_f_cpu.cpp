#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct DenseCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _b = oc->inputs[2];
        _y = oc->outputs[0];

        _w->Allocate();
        _b->Allocate();
        _y->Allocate();

        _m = _x->Dim(0);
        _n = _w->Dim(0);
        _k = _x->Dim(1);

        _bias_multiplier.Reshape({ _m });
        _bias_multiplier.Allocate();

        math::set<T, D>(
            _bias_multiplier.Size(),
            static_cast<T>(1),
            _bias_multiplier.GetPtr<T>()
            );
    }

    void Run() override {
        /*
        * Forward computation.
        * _x(batch_size x input_size) * _w(output_size x input_size)^T
        *  = _y(batch_size x output_size)
        */
        math::gemm<T, D>(
            false, true,
            _m, _n, _k,
            T(1), _x->GetPtr<T>(), _k,
            _w->GetPtr<T>(), _k,
            T(0), _y->GetPtr<T>(), _n, nullptr
            );

        /*
        * Add the bias term.
        * _y = _y + _b;
        */

        math::gemm<T, D>(
            false, false,
            _m, _n, 1,
            T(1), _bias_multiplier.GetPtr<T>(), 1
            , _b->GetPtr<T>(), _n,
            T(1), _y->GetPtr<T>(), _n, nullptr
            );
    }

    Tensor *_x, *_w, *_b;
    Tensor *_y;
    Tensor _bias_multiplier;
    int _m, _n, _k;
};

REGIST_NODE_FUNCTOR(Dense, DataType::F32, Accelerator::Default, DenseCpuF<float>)
REGIST_NODE_FUNCTOR(Dense, DataType::F64, Accelerator::Default, DenseCpuF<double>)

template <typename T, typename D = CPUContext>
struct DenseGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        _x = oc->inputs[0];
        _w = oc->inputs[1];
        _dy = oc->inputs[2];
        _dw = oc->outputs[0];
        _db = oc->outputs[1];
        _dx = oc->outputs[2];
        
        _dw->Allocate();
        _db->Allocate();
        _dx->Allocate();

        _m = _x->Dim(0);
        _n = _w->Dim(0);
        _k = _x->Dim(1);

        _bias_multiplier.Reshape({ _m });
        _bias_multiplier.Allocate();

        math::set<T, D>(
            _bias_multiplier.Size(),
            static_cast<T>(1),
            _bias_multiplier.GetPtr<T>()
            );
    }

    void Run() override {
        /*
        * _db = _dy.
        */
        math::gemv<T, D>(true, _m, _n, T(1),
            _dy->GetPtr<T>(), _n,
            _bias_multiplier.GetPtr<T>(), T(0),
            _db->GetPtr<T>(), _n, nullptr);

        /*
        * Calculate gradients of weights.
        * _dy(batch_size x output_size)^T * _x(batch_size x input_size)
        *  = _dw(output_size x input_size)
        */
        math::gemm<T, D>(true, false,
            _n, _k, _m,
            T(1), _dy->GetPtr<T>(), _n,
            _x->GetPtr<T>(), _k,
            T(0), _dw->GetPtr<T>(), _k, nullptr);

        /*
        * Calculate loss to propagate through bottom.
        * _dy(batch_size x output_size) * _w(output_size x input_size)
        *  = dx(batch_size x input_size)
        */
        math::gemm<T, D>(
            false, false,
            _m, _k, _n,
            T(1), _dy->GetPtr<T>(), _n,
            _w->GetPtr<T>(), _k,
            T(0), _dx->GetPtr<T>(), _k, nullptr);
    }

    Tensor *_x, *_w, *_dy;
    Tensor *_dw, *_db, *_dx;
    Tensor _bias_multiplier;
    int _m, _n, _k;
};

REGIST_NODE_GRADIENT_FUNCTOR(Dense, DataType::F32, Accelerator::Default, DenseGradCpuF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(Dense, DataType::F64, Accelerator::Default, DenseGradCpuF<double>)

} // end namespace node
} // end namespace mlfe