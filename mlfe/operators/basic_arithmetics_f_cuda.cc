#include "../core/op_algo.h"
#include "../core/device.h"
#include "../core/tensor_mem_ref.h"
#include "../math/basic_functions.h"
#include "../device_context/cuda_context.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Dev, class Tp>
class ElementwiseAdd : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseAdd(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        y = oac->get_output(0);
        size = y->Size();
    }

    void Compute() override{
        math::AddCuda<T>(size, x1->Data<T>(), x2->Data<T>(), y->Data<T>());
    }

private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(ElementwiseAdd)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseAdd<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ElementwiseAddGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseAddGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        dy = oac->get_input(2);
        dx1 = oac->get_output(0);
        dx2 = oac->get_output(1);
        size = dy->Size();
    }

    void Compute() override{
        auto copy_fn = Device::Copy<Device::CUDA, Device::CUDA>;
        copy_fn(dy->GetDeviceMemory(), dx1->GetDeviceMemory());
        copy_fn(dy->GetDeviceMemory(), dx2->GetDeviceMemory());
    }

private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *dy;
    TensorMemRef *dx1;
    TensorMemRef *dx2;
    int size;
};

REGIST_OP_GRAD_ALGO(ElementwiseAdd)
    .Input("dY", type::float32::string)
    .Output("dXs", "float32s")
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ElementwiseAddGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();


template <class Dev, class Tp>
class ElementwiseMul : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseMul(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        y = oac->get_output(0);
        size = y->Size();
    }

    void Compute() override{
        math::MulCuda<T>(size, x1->Data<T>(), x2->Data<T>(), y->Data<T>());
    }
private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *y;
    int size;
};

REGIST_OP_ALGO(ElementwiseMul)
    .Input("Xs", "float32s")
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseMul<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class ElementwiseMulGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseMulGrad(OpAlgoContext *oac) : OpAlgo(oac){
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        dy = oac->get_input(2);
        dx1 = oac->get_output(0);
        dx2 = oac->get_output(1);
        size = dy->Size();
    }

    void Compute() override{
        auto copy_fn = Device::Copy<Device::CUDA, Device::CUDA>;
        copy_fn(x2->GetDeviceMemory(), dx1->GetDeviceMemory());
        copy_fn(x1->GetDeviceMemory(), dx2->GetDeviceMemory());
        math::MulCuda<T>(size, dy->Data<T>(), dx1->Data<T>(), dx1->Data<T>());
        math::MulCuda<T>(size, dy->Data<T>(), dx2->Data<T>(), dx2->Data<T>());
    }

private:
    TensorMemRef *x1;
    TensorMemRef *x2;
    TensorMemRef *dy;
    TensorMemRef *dx1;
    TensorMemRef *dx2;
    int size;
};

REGIST_OP_GRAD_ALGO(ElementwiseMul)
    .Input("Xs", "float32s")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dXs", "float32s")
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ElementwiseMulGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class AddN : public OpAlgo{
using T = typename Tp::T;
public:
    AddN(OpAlgoContext *oac) : OpAlgo(oac){
        _num_inputs = oac->num_inputs();
        for(int n = 0; n < _num_inputs; ++n){
            xs.push_back(oac->get_input(n));
        }
        y = oac->get_output(0);
        size = xs[0]->Size();

    }

    void Compute() override{
        auto y_ptr = y->Data<T>();
        math::set<T, CUDAContext>(size, 0, y_ptr);
        for(int n = 0; n < _num_inputs; ++n){
            math::axpy<T, CUDAContext>(size, 1.f, xs[n]->Data<T>(), y_ptr);
        }
    }
private:
    std::vector<TensorMemRef *> xs;
    TensorMemRef *y;
    int size;
    int _num_inputs;
};

REGIST_OP_ALGO(AddN)
    .Input("Xs", "float32s")
    .Input("dy", "float32")
    .Output("Y", type::float32::string)
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = AddN<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Dev, class Tp>
class AddNGrad : public OpAlgo{
using T = typename Tp::T;
public:
    AddNGrad(OpAlgoContext *oac) : OpAlgo(oac){
        _num_inputs = oac->num_inputs();
        _num_outputs = oac->num_outputs();
        for(int n = 0; n < _num_inputs - 1; ++n){
            xs.push_back(oac->get_input(n));
        }
        for(int n = 0; n < _num_outputs; ++n){
            dxs.push_back(oac->get_output(n));
        }
        y = oac->get_input(_num_inputs - 2);
        dy = oac->get_input(_num_inputs - 1);
        size = dxs[0]->Size();
    }

    void Compute() override{
        auto dy_ptr = dy->Data<T>();
        for(int n = 0; n < _num_outputs; ++n){
            math::DivCuda<T>(size, y->Data<T>(), xs[n]->Data<T>(), dxs[n]->Data<T>());
            math::MulCuda<T>(size, dy_ptr, dxs[n]->Data<T>(), dxs[n]->Data<T>());
        }
    }

private:
    std::vector<TensorMemRef *> xs;
    std::vector<TensorMemRef *> dxs;
    TensorMemRef *y;
    TensorMemRef *dy;
    int size;
    int _num_inputs;
    int _num_outputs;
};

REGIST_OP_GRAD_ALGO(AddN)
    .Input("dY", "float32")
    .Output("dXs", "float32s")
    .Device(Device::CUDA::string)
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = AddNGrad<Device::CUDA, type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
