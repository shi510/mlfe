#include "../core/op_algo.h"
#include "../core/tensor_mem_ref.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cpu_context.h"
#include <cmath>
#include <algorithm>

namespace mlfe{ namespace algorithm_cpu{

template <class Dev, class Tp>
class SigmoidCrossEntropy : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropy(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        t = oac->get_input(1);
        loss = oac->get_output(0);

        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto loss_ptr = loss->Data<T>();

        for (int t = 0; t < m; ++t){
            loss_ptr[t] = T(0);
            for (int u = 0; u < n; ++u){
                int idx = t * n + u;
                T a = x_ptr[idx] * t_ptr[idx] - std::max(x_ptr[idx], T(0));
                T b = std::log(T(1) + std::exp(-std::abs(x_ptr[idx])));
                loss_ptr[t] += (a - b);
            }
            loss_ptr[t] = -loss_ptr[t] / static_cast<float>(n);
        }
    }
private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *loss;
    int m, n;
    int size;
};

REGIST_OP_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropy<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SigmoidCrossEntropyGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SigmoidCrossEntropyGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        t = oac->get_input(1);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();

        for(int b = 0; b < m; ++b){
            T dy_val = -dy_ptr[b] / T(n);
            for(int u = 0; u < n; ++u){
                int idx = b * n + u;
                T sig = T(1) / (T(1) + std::exp(-x_ptr[idx]));
                dx_ptr[idx] = (t_ptr[idx] - sig) * dy_val;
            }
        }
    }

private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *dy;
    TensorMemRef *dx;
    int m, n;
    int size;
};

REGIST_OP_GRAD_ALGO(SigmoidCrossEntropy)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SigmoidCrossEntropyGrad<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SoftmaxCrossEntropyWithLabel : public OpAlgo{
    using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabel(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        t = oac->get_input(1);
        loss = oac->get_output(0);

        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;

        prob = oac->GetDevice().CreateDeviceMemory();
        prob.Allocate(m * n * Tp::size);

        sm = oac->GetDevice().CreateDeviceMemory();
        sm.Allocate(n * Tp::size);

        rm = oac->GetDevice().CreateDeviceMemory();
        rm.Allocate(m * Tp::size);

        scal = oac->GetDevice().CreateDeviceMemory();
        scal.Allocate(m * Tp::size);

        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto loss_ptr = loss->Data<T>();
        auto sm_ptr = sm.Data<T>();
        auto rm_ptr = rm.Data<T>();
        auto scal_ptr = scal.Data<T>();
        auto prob_ptr = prob.Data<T>();

        math::rowwise_max<T, CPUContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        math::scal<T, CPUContext>(
            m * n, T(1),
            x_ptr,
            prob_ptr
            );

        math::gemm<T, CPUContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, nullptr);

        math::exp<T, CPUContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CPUContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, nullptr);

        math::rowwise_normalize<T, CPUContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy<T, CPUContext>(m, n,
            prob_ptr,
            t_ptr,
            loss_ptr
            );
    }
private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *loss;
    DeviceMemory sm;
    DeviceMemory rm;
    DeviceMemory scal;
    DeviceMemory prob;
    int m, n;
    int size;
};

REGIST_OP_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Output("Loss", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabel<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class SoftmaxCrossEntropyWithLabelGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SoftmaxCrossEntropyWithLabelGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x = oac->get_input(0);
        t = oac->get_input(1);
        dy = oac->get_input(2);
        dx = oac->get_output(0);
        m = x->Shape()[0];
        n = x->Shape()[1];
        size = m * n;

        prob = oac->GetDevice().CreateDeviceMemory();
        prob.Allocate(m * n * Tp::size);

        sm = oac->GetDevice().CreateDeviceMemory();
        sm.Allocate(n * Tp::size);

        rm = oac->GetDevice().CreateDeviceMemory();
        rm.Allocate(m * Tp::size);

        scal = oac->GetDevice().CreateDeviceMemory();
        scal.Allocate(m * Tp::size);

        math::set<T, CPUContext>(
            n,
            static_cast<T>(1),
            sm.Data<T>()
            );
    }

    void Compute() override{
        auto x_ptr = x->Data<T>();
        auto t_ptr = t->Data<T>();
        auto dy_ptr = dy->Data<T>();
        auto dx_ptr = dx->Data<T>();
        auto sm_ptr = sm.Data<T>();
        auto rm_ptr = rm.Data<T>();
        auto scal_ptr = scal.Data<T>();
        auto prob_ptr = prob.Data<T>();

        math::rowwise_max<T, CPUContext>(
            m, n,
            x_ptr,
            rm_ptr
            );

        math::scal<T, CPUContext>(
            m * n, T(1),
            x_ptr,
            prob_ptr
            );

        math::gemm<T, CPUContext>(false, false,
            m, n, 1,
            T(-1), rm_ptr, 1,
            sm_ptr, n,
            T(1), prob_ptr, n, nullptr);

        math::exp<T, CPUContext>(
            m * n,
            prob_ptr,
            prob_ptr
            );

        math::gemv<T, CPUContext>(false,
            m, n,
            T(1), prob_ptr, n,
            sm_ptr,
            T(0), scal_ptr, 1, nullptr);

        math::rowwise_normalize<T, CPUContext>(m, n,
            scal_ptr,
            prob_ptr
            );

        math::cross_entropy_gradient<T, CPUContext>(
            m, n,
            prob_ptr,
            t_ptr,
            dy_ptr,
            dx_ptr
            );
    }

private:
    TensorMemRef *x;
    TensorMemRef *t;
    TensorMemRef *dy;
    TensorMemRef *dx;
    DeviceMemory sm;
    DeviceMemory rm;
    DeviceMemory scal;
    DeviceMemory prob;
    int m, n;
    int size;
};

REGIST_OP_GRAD_ALGO(SoftmaxCrossEntropyWithLabel)
    .Input("X", type::float32::string)
    .Input("Target", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device(Device::CPU::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SoftmaxCrossEntropyWithLabelGrad<Device::CPU, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
