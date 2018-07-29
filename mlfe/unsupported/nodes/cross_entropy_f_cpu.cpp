#include "../core/node.hpp"
#include "../../device_context/cpu_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions.hpp"
#include <cmath>

namespace mlfe { namespace node {

template <typename T, typename D = CPUContext>
struct SoftmaxXentWithLabelCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _x = oc->inputs[0];
        _label = oc->inputs[1];
        _prob = oc->outputs[0];
        _loss = oc->outputs[1];
        _m = _x->Dim(0);
        _n = _x->Dim(1);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _prob->Allocate(Accelerator::Default, dt);
        _loss->Allocate(Accelerator::Default, dt);

        _sum_multiplier.Reshape({ _n });
        _sum_multiplier.Allocate(Accelerator::Default, dt);
        _rowwise_max.Reshape({ _m });
        _rowwise_max.Allocate(Accelerator::Default, dt);
        _scaler.Reshape({ _m });
        _scaler.Allocate(Accelerator::Default, dt);

        math::set<T, D>(
            _sum_multiplier.Size(),
            static_cast<T>(1),
            _sum_multiplier.GetPtr<T>()
            );
    }

    void Run() override {
        math::rowwise_max<T, D>(
            _m, _n,
            _x->GetPtr<T>(),
            _rowwise_max.GetPtr<T>()
            );

        math::scal<T, D>(
            _m * _n, T(1),
            _x->GetPtr<T>(),
            _prob->GetPtr<T>()
            );

        math::gemm<T, D>(false, false,
            _m, _n, 1,
            T(-1), _rowwise_max.GetPtr<T>(), 1,
            _sum_multiplier.GetPtr<T>(), _n,
            T(1), _prob->GetPtr<T>(), _n, nullptr);

        math::exp<T, D>(
            _prob->Size(),
            _prob->GetPtr<T>(),
            _prob->GetPtr<T>()
            );

        math::gemv<T, D>(false,
            _m, _n,
            T(1), _prob->GetPtr<T>(), _n,
            _sum_multiplier.GetPtr<T>(),
            T(0), _scaler.GetPtr<T>(), 1, nullptr);

        math::rowwise_normalize<T, D>(_m, _n,
            _scaler.GetPtr<T>(),
            _prob->GetPtr<T>()
            );

        math::cross_entropy<T, D>(_m, _n,
            _prob->GetPtr<T>(),
            _label->GetPtr<T>(),
            _rowwise_max.GetPtr<T>()
            );

        math::sum<T, D>(
            _m,
            _rowwise_max.GetPtr<T>(),
            _loss->GetPtr<T>()
            );

        math::scal<T, D>(
            1,
            static_cast<T>(1) / static_cast<T>(_m),
            _loss->GetPtr<T>(),
            _loss->GetPtr<T>()
            );
    }

    Tensor *_x, *_label;
    Tensor *_prob, *_loss;
    Tensor _rowwise_max;
    Tensor _scaler;
    Tensor _sum_multiplier;
    int _m, _n;
};

REGIST_NODE_FUNCTOR(SoftmaxXentWithLabel, DataType::F32, Accelerator::Default, SoftmaxXentWithLabelCpuF<float>)
REGIST_NODE_FUNCTOR(SoftmaxXentWithLabel, DataType::F64, Accelerator::Default, SoftmaxXentWithLabelCpuF<double>)

template <typename T, typename D = CPUContext>
struct SoftmaxXentWithLabelGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _label = oc->inputs[0];
        _prob = oc->inputs[1];
        _loss = oc->inputs[2];
        _dx = oc->outputs[0];

        _m = _prob->Dim(0);
        _n = _prob->Dim(1);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dx->Allocate(Accelerator::Default, dt);
    }
    void Run() override {
        math::cross_entropy_gradients<T, D>(_m, _n,
            _prob->GetPtr<T>(),
            _label->GetPtr<T>(),
            _loss->GetPtr<T>(),
            _dx->GetPtr<T>()
            );
    }

    Tensor *_label, *_prob, *_loss;
    Tensor *_dx;
    int _m, _n;
};

REGIST_NODE_GRADIENT_FUNCTOR(SoftmaxXentWithLabel, DataType::F32, Accelerator::Default, SoftmaxXentWithLabelGradCpuF<float>)
REGIST_NODE_GRADIENT_FUNCTOR(SoftmaxXentWithLabel, DataType::F64, Accelerator::Default, SoftmaxXentWithLabelGradCpuF<double>)

template <typename T, typename D = CPUContext>
struct SigmoidXentCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _x = oc->inputs[0];
        _sig = oc->inputs[1];
        _t = oc->inputs[2];
        _loss = oc->outputs[0];
        _m = _x->Dim(0);
        _n = _x->Dim(1);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _loss->Allocate(Accelerator::Default, dt);
        _sig->Allocate(Accelerator::Default, dt);
    }

    void Run() override {
        const T *x_ptr = _x->GetPtr<T>();
        T *sig_ptr = _sig->GetPtr<T>();
        T *t_ptr = _t->GetPtr<T>();
        T *loss_ptr = _loss->GetPtr<T>();
        math::set<T, D>(_loss->Size(), static_cast<T>(0), loss_ptr);

        math::SigmoidFunction<T, D>(
            _x->Size(),
            x_ptr,
            sig_ptr
            );

        for (int t = 0; t < _m; ++t) {
            for (int u = 0; u < _n; ++u) {
                loss_ptr[t] -= (t_ptr[u] - static_cast<float>(1)) * sig_ptr[u] - std::log(1 + std::exp(-sig_ptr[u]));
            }
            loss_ptr[t] /= static_cast<float>(_n);
        }
    }

    Tensor *_x, *_sig, *_t;
    Tensor *_loss;
    int _m, _n;
};

REGIST_NODE_FUNCTOR(SigmoidXent, DataType::F32, Accelerator::Default, SigmoidXentCpuF<float>)
//REGIST_NODE_FUNCTOR(SigmoidXent, DataType::F64, Accelerator::Default, SigmoidXentCpuF<double>)

template <typename T, typename D = CPUContext>
struct SigmoidXentGradCpuF : NodeFunctor {
    void Init(OperatorContext *oc) override {
        auto dt = DataType::F32;
        _x = oc->inputs[0];
        _sig = oc->inputs[1];
        _t = oc->inputs[2];
        _loss = oc->inputs[3];
        _dx = oc->outputs[0];
        _m = _x->Dim(0);
        _n = _x->Dim(1);

        // TODO : not use type size compare.
        if (sizeof(T) == 8) {
            dt = DataType::F64;
        }
        _dx->Allocate(Accelerator::Default, dt);
    }

    void Run() override {
        const T *sig_ptr = _sig->GetPtr<T>();
        const T *t_ptr = _t->GetPtr<T>();
        const T *loss_ptr = _loss->GetPtr<T>();
        T *dx_ptr = _dx->GetPtr<T>();

        for (int t = 0; t < _m; ++t) {
            T scaler = loss_ptr[t] / static_cast<T>(_n);
            for (int u = 0; u < _n; ++u) {
                dx_ptr[t * _n + u] = (sig_ptr[t * _n + u] - t_ptr[t * _n + u]) * scaler;
            }
        }
    }

    Tensor *_x, *_sig, *_t, *_loss;
    Tensor *_dx;
    int _m, _n;
};

REGIST_NODE_GRADIENT_FUNCTOR(SigmoidXent, DataType::F32, Accelerator::Default, SigmoidXentGradCpuF<float>)
//REGIST_NODE_GRADIENT_FUNCTOR(SigmoidXent, DataType::F64, Accelerator::Default, SigmoidXentGradCpuF<double>)

} // end namespace node
} // end namespace mlfe
