#include "mlfe/core/op_algo.h"
#include "mlfe/core/device.h"
#include "mlfe/math/basic_functions.h"
#include "mlfe/device_context/cuda_context.h"
#include "mlfe/math/blas.h"

namespace mlfe{
namespace algorithm_cuda{

template <class Tp>
class Negative : public OpAlgo{
using T = typename Tp::T;
public:
    Negative(OpAlgoContext *oac) : OpAlgo(oac, "Negative"){
        y = oac->get_output(0);
        x = y.get_children()[0];
        size = y.size();
    }

    void Compute() override{
        auto x_ptr = x.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::negative<float, CUDAContext>(size, x_ptr, y_ptr);
    }

private:
    Tensor x;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(Negative)
    .Input("X", "float32")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = Negative<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

#define ADD_BASIC_OP(Name)                                           \
template <class Tp>                                                  \
class Elementwise##Name : public OpAlgo{                             \
using T = typename Tp::T;                                            \
public:                                                              \
    Elementwise##Name(OpAlgoContext *oac)                            \
        : OpAlgo(oac, "Elementwise" # Name){                         \
        y = oac->get_output(0);                                      \
        x1 = y.get_children()[0];                                    \
        x2 = y.get_children()[1];                                    \
        size = y.size();                                             \
    }                                                                \
    void Compute() override{                                         \
        auto x1_ptr = x1.device_data<T>();                           \
        auto x2_ptr = x2.device_data<T>();                           \
        auto y_ptr = y.mutable_device_data<T>();                     \
        math::Name##Cuda<T>(size, x1_ptr, x2_ptr, y_ptr);            \
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
    .Device("CUDA")                                                  \
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{    \
        using T = Elementwise##Name<type::float32>;                  \
        return std::make_shared<T>(oac);                             \
    })                                                               \
    .Finish();

ADD_BASIC_OP(Add)
ADD_BASIC_OP(Sub)
ADD_BASIC_OP(Mul)
ADD_BASIC_OP(Div)

#undef ADD_BASIC_OP

#if 0
template <class Tp>
class ElementwiseAdd : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseAdd(OpAlgoContext *oac) : OpAlgo(oac, "ElementwiseAdd"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = y.size();
    }

    void Compute() override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::AddCuda<T>(size, x1_ptr, x2_ptr, y_ptr);
    }

private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ElementwiseAdd)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseAdd<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ElementwiseSub : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseSub(OpAlgoContext *oac) : OpAlgo(oac, "ElementwiseSub"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = y.size();
    }

    void Compute() override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::SubCuda<T>(size, x1_ptr, x2_ptr, y_ptr);
    }

private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ElementwiseSub)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseSub<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ElementwiseMul : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseMul(OpAlgoContext *oac) : OpAlgo(oac, "ElementwiseMul"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = y.size();
    }

    void Compute() override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::MulCuda<T>(y.size(), x1_ptr, x2_ptr, y_ptr);
    }

private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ElementwiseMul)
    .Input("Xs", "float32s")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseMul<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class ElementwiseDiv : public OpAlgo{
using T = typename Tp::T;
public:
    ElementwiseDiv(OpAlgoContext *oac) : OpAlgo(oac, "ElementwiseDiv"){
        y = oac->get_output(0);
        x1 = y.get_children()[0];
        x2 = y.get_children()[1];
        size = y.size();
    }

    void Compute() override{
        auto x1_ptr = x1.device_data<T>();
        auto x2_ptr = x2.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        math::DivCuda<T>(y.size(), x1_ptr, x2_ptr, y_ptr);
    }

private:
    Tensor x1;
    Tensor x2;
    Tensor y;
    int size;
};

REGIST_OP_ALGO(ElementwiseDiv)
    .Input("Xs", "float32s")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = ElementwiseDiv<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

#endif

template <class Tp>
class AddN : public OpAlgo{
using T = typename Tp::T;
public:
    AddN(OpAlgoContext *oac) : OpAlgo(oac, "AddN"){
        y = oac->get_output(0);
        size = y.size();
        num_inputs = y.get_children().size();
        for(int n = 0; n < num_inputs; ++n){
            xs.push_back(y.get_children()[n]);
        }
    }

    void Compute() override{
        auto y_ptr = y.mutable_device_data<T>();
        math::set<T, CUDAContext>(size, 0, y_ptr);
        for(int n = 0; n < num_inputs; ++n){
            auto x_ptr = xs[n].device_data<T>();
            math::axpy<T, CUDAContext>(size, 1.f, x_ptr, y_ptr);
        }
    }

private:
    Tensor y;
    std::vector<Tensor> xs;
    int size;
    int num_inputs;
};

REGIST_OP_ALGO(AddN)
    .Input("Xs", "float32s")
    .Input("dy", "float32")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = AddN<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

template <class Tp>
class MatrixVectorAdd : public OpAlgo{
using T = typename Tp::T;
public:
    MatrixVectorAdd(OpAlgoContext *oac) : OpAlgo(oac, "MatrixVectorAdd"){
        y = oac->get_output(0);
        mat = y.get_children()[0];
        vec = y.get_children()[1];
        m = mat.shape()[0];
        n = mat.shape()[1];
        multiplier = create_memory(m * Tp::size);
        math::set<T, CUDAContext>(m, T(1), 
                                  multiplier->mutable_device_data<T>());
    }

    void Compute() override{
        auto mat_ptr = mat.device_data<T>();
        auto vec_ptr = vec.device_data<T>();
        auto y_ptr = y.mutable_device_data<T>();
        auto mul_ptr = multiplier->device_data<T>();

        copy(mat.get_memory(), y.get_memory());

        math::gemm<T, CUDAContext>(
            false, false,
            m, n, 1,
            T(1), mul_ptr, 1,
            vec_ptr, n,
            T(1), y_ptr, n, &cxt
            );
    }

private:
    Tensor mat;
    Tensor vec;
    Tensor y;
    memory_ptr multiplier;
    CUDAContext cxt;
    int m, n;
};

REGIST_OP_ALGO(MatrixVectorAdd)
    .Input("Mat", "float32")
    .Input("Vec", "float32")
    .Output("Y", type::float32::string)
    .Device("CUDA")
    .CreatorFn([](OpAlgoContext *oac) -> std::shared_ptr<OpAlgo>{
        using T = MatrixVectorAdd<type::float32>;
        return std::make_shared<T>(oac);
    })
    .Finish();

} // end namespace algorithm_cuda
} // end namespace mlfe
