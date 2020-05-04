#include "../core/op_algo.h"
#include "../math/blas.h"
#include "../math/basic_functions.h"
#include "../math/activations.h"
#include "../device_context/cuda_context.h"
#include "../core/device.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class SquaredDifference : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifference(OpAlgoContext *oac) : OpAlgo(oac, "SquaredDifference"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = x1.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::squared_difference<T, CUDAContext>(size, x1_ptr, x2_ptr, y_ptr);
    }

private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifference<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class SquaredDifferenceGrad : public OpAlgo{
using T = typename Tp::T;
public:
    SquaredDifferenceGrad(OpAlgoContext *oac) : OpAlgo(oac){}

    void Compute(op_algo_runtime_context& rc) override{}

private:
    Tensor x1;
    Tensor x2;
    Tensor dy;
    Tensor dx1;
    Tensor dx2;
    int size;
};

REGIST_OP_GRAD_ALGO(SquaredDifference)
    .Input("X1", type::float32::string)
    .Input("X2", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX1", type::float32::string)
    .Output("dX2", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = SquaredDifferenceGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReduceMean : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMean(OpAlgoContext *oac) : OpAlgo(oac, "ReduceMean"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        size = x.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::set<T, CUDAContext>(1, T(0), y_ptr);
        math::sum<T, CUDAContext>(size, x_ptr, y_ptr);
        math::scal<T, CUDAContext>(1, T(1) / T(size), y_ptr, y_ptr);
    }

private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMean<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ReduceMeanGrad : public OpAlgo{
using T = typename Tp::T;
public:
    ReduceMeanGrad(OpAlgoContext *oac) : OpAlgo(oac, "ReduceMeanGradient"){
        dx = oac->get_output(0);
        dy = dx.get_children()[0];
        size = dx.size();
        scale = T(1) / T(size);
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto dy_ptr = dy.device_data<T>();
        auto dx_ptr = dx.mutable_device_data<T>();
        math::reduce_mean_gradient<T, CUDAContext>(size,
                                                   scale,
                                                   dy_ptr,
                                                   dx_ptr
                                                  );
    }

private:
    Tensor dy;
    Tensor dx;
    int size;
    T scale;
};

REGIST_OP_GRAD_ALGO(ReduceMean)
    .Input("X", type::float32::string)
    .Input("dY", type::float32::string)
    .Output("dX", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) ->std::shared_ptr<OpAlgo>{
        using T = ReduceMeanGrad<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
