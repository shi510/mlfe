#include "../core/node.hpp"
#include "../../device_context/cuda_context.hpp"
#include "../../math/blas.hpp"
#include "../../math/functions_cuda.hpp"

namespace mlfe { namespace node {
//TODO : 
// make more simple on the two cases that are: 
// 1. one input, one value and one output. (C_Tensor = A_Tensor [BinaryOpExpr] Constant_Value)
// 2. two input and one output. (C_Tensor = A_Tensor [BinaryOpExpr] B_Tensor)
#define DEFINE_BIANRY_OP_NODE_FUNCTOR(OpName)\
template <typename T, typename D = CUDAContext>\
struct OpName##CudaF : NodeFunctor {\
    void Init(OperatorContext *oc) override {\
        _a = oc->inputs[0];\
        _c = oc->outputs[0];\
        _b = nullptr;\
        if(oc->inputs.size() > 1){\
            _b = oc->inputs[1];\
        }\
        else{\
            _val = to_value<T>(oc->attr->GetParam<std::string>("Value"));\
        }\
\
        _size = _c->Size();\
    }\
    void Run() override {\
        const T *a_ptr = _a->GetPtr<T>();\
        T *c_ptr = _c->GetPtr<T>();\
        if(_b != nullptr){\
            const T *b_ptr = _b->GetPtr<T>();\
            math::##OpName##Cuda<T>(_size, a_ptr, b_ptr, c_ptr);\
        }\
        else {\
            math::##OpName##ValCuda<T>(_size, _val, a_ptr, c_ptr);\
        }\
    }\
    Tensor *_a, *_b;\
    Tensor *_c;\
    T _val;\
    int _size;\
};

DEFINE_BIANRY_OP_NODE_FUNCTOR(Add)
REGIST_NODE_FUNCTOR(Add, DataType::F32, Accelerator::CUDA, AddCudaF<float>)
REGIST_NODE_FUNCTOR(Add, DataType::F64, Accelerator::CUDA, AddCudaF<double>)

DEFINE_BIANRY_OP_NODE_FUNCTOR(Sub)
REGIST_NODE_FUNCTOR(Sub, DataType::F32, Accelerator::CUDA, SubCudaF<float>)
REGIST_NODE_FUNCTOR(Sub, DataType::F64, Accelerator::CUDA, SubCudaF<double>)

DEFINE_BIANRY_OP_NODE_FUNCTOR(Mul)
REGIST_NODE_FUNCTOR(Mul, DataType::F32, Accelerator::CUDA, MulCudaF<float>)
REGIST_NODE_FUNCTOR(Mul, DataType::F64, Accelerator::CUDA, MulCudaF<double>)

DEFINE_BIANRY_OP_NODE_FUNCTOR(Div)
REGIST_NODE_FUNCTOR(Div, DataType::F32, Accelerator::CUDA, DivCudaF<float>)
REGIST_NODE_FUNCTOR(Div, DataType::F64, Accelerator::CUDA, DivCudaF<double>)

} // end namespace node
} // end namespace mlfe