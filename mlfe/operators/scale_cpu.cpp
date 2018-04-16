#include "scale.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DT, class DC>
ScaleOp<DT, DC>::ScaleOp(
                                    OperatorIO &opio,
                                    ItemHolder *ih
                                    ) : Operator<DC>(opio, ih) {
    runtime_assert(this->inputs.size() == 1,
                   "[Scale Op] inputs.size() == 1.");
    runtime_assert(this->outputs.size() == 1,
                   "[Scale Op] outputs.size() == 1.");
    const auto x = this->inputs[InputSchema::x];
    auto y = this->outputs[OutputSchema::y];
    
    runtime_assert(opio.param.HasParam("Scale"),
                   "[Scale Op] Not found Scale param.");
    if(y->IsEmpty() &&
       !x->IsEmpty()
       ){
        scaler = opio.param.GetParam<DT>("Scale");
        y->template Resize<DT>(*x);
    }
    else{
        runtime_assert(x->Dims() == y->Dims(),
                       "[Scale Op] x->Dims() == y->Dims().");
    }
}

template <class DT, class DC>
void ScaleOp<DT, DC>::Compute(){
    const auto x = this->inputs[InputSchema::x];
    auto y = this->outputs[OutputSchema::y];
    math::scal<DT, DC>(
                                  x->Size(),
                                  scaler,
                                  x->template GetPtrConst<DT>(),
                                  y->template GetPtrMutable<DT>()
                                  );
}


REGIST_OPERATOR_CPU(Scale_float, ScaleOp<float, CPUContext>)
REGIST_OPERATOR_CPU(Scale_double, ScaleOp<double, CPUContext>)

struct ScaleGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type;
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param.Add("Scale", 1.f);
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(Scale, ScaleGradientIO)

} /* namespace mlfe */
