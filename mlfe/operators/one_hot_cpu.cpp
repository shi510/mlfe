#include "one_hot.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <>
OneHotOp<float, CPUContext>::OneHotOp(
                                      OperatorIO &opio,
                                      ItemHolder *ih
                                      ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 1,
                   "[OneHot Op] inputs.size() == 1.");
    runtime_assert(outputs.size() == 1,
                   "[OneHot Op] outputs.size() == 1.");
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    
    if(opio.param.HasParam("Dim") &&
       y->IsEmpty() &&
       !x->IsEmpty()
       ){
        dim = opio.param.GetParam<int>("Dim");
        y->Resize<float>({x->Dim(0), dim});
    }
    else{
        runtime_assert(x->Dims() == y->Dims(),
                       "[OneHot Op] x->Dims() == y->Dims().");
    }
}

template <>
void OneHotOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    auto y = outputs[OutputSchema::y];
    math::scal<float, CPUContext>(
                                  y->Size(),
                                  static_cast<float>(0),
                                  y->GetPtrConst<float>(),
                                  y->GetPtrMutable<float>()
                                  );
    for(int b = 0; b < x->Dim(0); ++b){
        int val = x->GetPtrConst<float>()[b];
        y->GetPtrMutable<float>()[b * dim + val] = static_cast<float>(1);
    }
}

REGIST_OPERATOR_CPU(OneHot, OneHotOp<float, CPUContext>)

} /* namespace mlfe */
