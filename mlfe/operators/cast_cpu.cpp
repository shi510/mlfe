#include "cast.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <>
CastOp<CPUContext>::CastOp(
                           OperatorIO &opio,
                           ItemHolder *ih
                           ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 1,
                   "[Cast Op] inputs.size() == 1.");
    runtime_assert(outputs.size() == 1,
                   "[Cast Op] inputs.size() == 1.");
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    runtime_assert(opio.param.HasParam("Cast"),
                   "[Cast Op] Not found Cast param.");
    if(y->IsEmpty() &&
       !x->IsEmpty()
       ){
        cast = opio.param.GetParam<std::string>("Cast");
        if(!cast.compare("char")){
            y->Resize<char>(*x);
        }
        else if(!cast.compare("int")){
            y->Resize<int>(*x);
        }
        else if(!cast.compare("float")){
            y->Resize<float>(*x);
        }
        else if(!cast.compare("double")){
            y->Resize<double>(*x);
        }
        else{
            throw std::string("Wrong Type.");
        }
    }
    else{
        runtime_assert(x->Dims() == y->Dims(),
                       "[Cast Op] x->Dims() == y->Dims().");
    }
}

template<>
void CastOp<CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    if(!cast.compare("char")){
        TypeCaster<TypeLists<char, unsigned char, int, float, double>, char>::Run(x, y);
    }
    else if(!cast.compare("int")){
        TypeCaster<TypeLists<char, unsigned char, int, float, double>, int>::Run(x, y);
    }
    else if(!cast.compare("float")){
        TypeCaster<TypeLists<char, unsigned char, int, float, double>, float>::Run(x, y);
    }
    else if(!cast.compare("double")){
        TypeCaster<TypeLists<char, unsigned char, int, float, double>, double>::Run(x, y);
    }
    else{
        throw std::string("Wrong Type. -> ") + std::string(cast);
    }
}

template <>
template <class From, class To>
void CastOp<CPUContext>::TypeCast(
    TensorBlob<CPUContext> *from,
    TensorBlob<CPUContext> *to
) {
    const From *from_ptr = from->template GetPtrConst<From>();
    To *to_ptr = to->template GetPtrMutable<To>();
    for (int n = 0; n < from->Size(); ++n) {
        to_ptr[n] = static_cast<To>(from_ptr[n]);
    }
}

REGIST_OPERATOR_CPU(Cast, CastOp<CPUContext>)

} /* namespace mlfe */
