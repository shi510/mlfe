#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/math/blas.h"
#include "mlfe/math/transform.h"
#include "mlfe/device_context/cpu_context.h"

namespace mlfe{
namespace algorithm_cpu{

template <class Tp>
class Negative : public OpAlgo{
using T = typename Tp::T;
public:
    Negative(OpAlgoContext *oac) : OpAlgo(oac, "Negative"){
        y = oac->get_output(0);
        x = oac->get_input(0);
        size = y.size();
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        for(int n = 0; n < size; ++n){
            y_ptr[n] = -x_ptr[n];
        }
    }

private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(Negative)
    .Input("X", "float32")
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Negative<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

#define ADD_BASIC_OP(Name, Expr)                                     \
template <class Tp>                                                  \
class Elementwise##Name : public OpAlgo{                             \
using T = typename Tp::T;                                            \
public:                                                              \
    Elementwise##Name(OpAlgoContext *oac)                            \
        : OpAlgo(oac, "Elementwise" # Name){                         \
        y = oac->get_output(0);                                      \
        x1 = oac->get_input(0);                                      \
        x2 = oac->get_input(1);                                      \
        size = y.size();                                             \
    }                                                                \
    void Compute(op_algo_runtime_context& rc) override{              \
        auto x1_ptr = x1.device_data<T>();                           \
        auto x2_ptr = x2.device_data<T>();                           \
        auto y_ptr = y.mutable_device_data<T>();                     \
        for(int n = 0; n < size; ++n){                               \
            y_ptr[n] = x1_ptr[n] Expr x2_ptr[n];                     \
        }                                                            \
    }                                                                \
private:                                                             \
    Tensor x1;                                                       \
    Tensor x2;                                                       \
    Tensor y;                                                        \
    int size;                                                        \
};                                                                   \
REGIST_OP_ALGO(Elementwise##Name)                                    \
    .Input("X1", "float32")                                          \
    .Input("X2", "float32")                                          \
    .Output("Y", type::float32::string)                              \
    .Device("CPU")                                                   \
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{    \
        using T = Elementwise##Name<type::float32>;                  \
        return std::make_shared<T>(oac);                             \
    })                                                               \
    .Finish();

ADD_BASIC_OP(Add, +)
ADD_BASIC_OP(Sub, -)
ADD_BASIC_OP(Mul, *)
ADD_BASIC_OP(Div, /)

#undef ADD_BASIC_OP

template <class Tp>
class AddWithBroadcast : public OpAlgo{
using T = typename Tp::T;
public:
    AddWithBroadcast(OpAlgoContext *oac)
        : OpAlgo(oac, "AddWithBroadcast"){
        y = oac->get_output(0);
        x1 = oac->get_input(0);
        x2 = oac->get_input(1);
        size = y.size();
        from_shape.resize(4);
        to_shape.resize(4);
        std::fill(from_shape.begin(), from_shape.end(), 1);
        std::fill(to_shape.begin(), to_shape.end(), 1);
        std::copy(x2.shape().begin(), x2.shape().end(), from_shape.begin());
        std::copy(y.shape().begin(), y.shape().end(), to_shape.begin());
    }
    
    void Compute(op_algo_runtime_context& rc) override {
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::broadcast<T, CPUContext>(x2_ptr, y_ptr,
            from_shape[0], from_shape[1], from_shape[2], from_shape[3],
            to_shape[0], to_shape[1], to_shape[2], to_shape[3]);

        for(int n = 0; n < size; ++n){
            y_ptr[n] = y_ptr[n] + x1_ptr[n];
        }
    }
private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    std::vector<int> from_shape;
    std::vector<int> to_shape;
    int size;
};
REGIST_OP_ALGO(AddWithBroadcast)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = AddWithBroadcast<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class AddWithBroadcastGrad : public OpAlgo {
    using T = typename Tp::T;
public:
    AddWithBroadcastGrad(OpAlgoContext* oac)
        : OpAlgo(oac, "AddWithBroadcastGrad") {
        dx1 = oac->get_output(0);
        dx2 = oac->get_output(1);
        dy = oac->get_input(0);
        x1 = oac->get_input(1);
        x2 = oac->get_input(2);
        y = oac->get_input(3);
        size = y.size();
        from_shape.resize(4);
        to_shape.resize(4);
        std::fill(from_shape.begin(), from_shape.end(), 1);
        std::fill(to_shape.begin(), to_shape.end(), 1);
        std::copy(x2.shape().begin(), x2.shape().end(), from_shape.begin());
        std::copy(dy.shape().begin(), dy.shape().end(), to_shape.begin());
    }
    void Compute(op_algo_runtime_context& rc) override {
        auto dx1_ptr = dx1.mutable_device_data<T>();
        auto dx2_ptr = dx2.mutable_device_data<T>();
        auto dy_ptr = dy.device_data<T>();
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.device_data<T>();

        copy(dy.get_memory(), dx1.get_memory());
        math::set<T, CPUContext>(dx2.size(), T(0), dx2_ptr);
        math::broadcast_gradient<T, CPUContext>(dy_ptr, dx2_ptr,
            to_shape[0], to_shape[1], to_shape[2], to_shape[3],
            from_shape[0], from_shape[1], from_shape[2], from_shape[3]);
    }
private:
    Tensor dx1;
    Tensor dx2;
    Tensor dy;
    Tensor x1;
    Tensor x2;
    Tensor y;
    std::vector<int> from_shape;
    std::vector<int> to_shape;
    int size;
};

REGIST_OP_GRAD_ALGO(AddWithBroadcast)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext* oac) -> std::shared_ptr<OpAlgo> {
    using T = AddWithBroadcastGrad<type::float32>;
    return std::make_shared<T>(oac);
        })
    .Finish();

template <class Tp>
class AddN : public OpAlgo{
using T = typename Tp::T;
public:
    AddN(OpAlgoContext *oac) : OpAlgo(oac, "AddN"){
        y = oac->get_output(0);
        size = y.size();
        _num_inputs = oac->num_inputs();
        for(int n = 0; n < _num_inputs; ++n){
            xs.push_back(oac->get_input(n));
        }
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto y_ptr = y.mutable_device_data<T>();
        math::set<T, CPUContext>(size, 0, y_ptr);
        for(auto &x : xs){
            auto x_ptr = x.template device_data<T>();
            math::axpy<T, CPUContext>(size, 1.f, x_ptr, y_ptr);
        }
    }
private:
    std::vector<Tensor> xs;
    Tensor y;
    int size;
    int _num_inputs;
};

REGIST_OP_ALGO(AddN)
    .Input("Xs", "float32s")
    .Input("dy", "float32")
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = AddN<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class MatrixVectorAdd : public OpAlgo{
using T = typename Tp::T;
public:
    MatrixVectorAdd(OpAlgoContext *oac) 
        : OpAlgo(oac, "MatrixVectorAdd"){
        y = oac->get_output(0);
        mat = oac->get_input(0);
        vec = oac->get_input(1);
        m = mat.shape()[0];
        n = mat.shape()[1];
        multiplier = create_memory(m * Tp::size);
        math::set<T, CPUContext>(m, T(1), 
                                 multiplier->mutable_device_data<T>());
    }

    void Compute(op_algo_runtime_context& rc) override{
        auto mat_ptr = mat.device_data<T>();
        auto vec_ptr = vec.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto mul_ptr = multiplier->device_data<T>();

        copy(mat.get_memory(), y.get_memory());

        math::gemm<T, CPUContext>(
            false, false,
            m, n, 1,
            T(1), mul_ptr, 1,
            vec_ptr, n,
            T(1), y_ptr, n, nullptr
            );
    }

private:
    Tensor mat;
    Tensor vec;
    Tensor y;
    memory_ptr multiplier;
    int m, n;
};

REGIST_OP_ALGO(MatrixVectorAdd)
    .Input("Mat", "float32")
    .Input("Vec", "float32")
    .Output("Y", type::float32::string)
    .Device("CPU")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MatrixVectorAdd<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cpu
} // end namespace mlfe
