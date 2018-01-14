#include "flatten.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DT, class DC>
FlattenOp<DT, DC>::FlattenOp(
                                        OperatorIO &opio,
                                        ItemHolder *ih
                                        ) : Operator<DC>(opio, ih) {
    runtime_assert(this->inputs.size() == 1,
                   "[Flatten Op] inputs.size() == 1.");
    runtime_assert(this->outputs.size() == 1,
                   "[Flatten Op] outputs.size() == 1.");
    const auto x = this->inputs[InputSchema::x];
    auto y = this->outputs[OutputSchema::y];
    
    runtime_assert(opio.param.HasParam("Axis"), "[Flatten Op] Not found Axis param.");
    if(y->IsEmpty() &&
       !x->IsEmpty()
       ){
        int flat_from = 1;
        int flat_to = 1;
        axis = opio.param.GetParam<int>("Axis");
        *y = *x;
        for(int n = 0; n < axis; ++n){
            flat_from *= x->Dim(n);
        }
        for(int n = x->Dims() - 1; n >= axis; --n){
            flat_to *= x->Dim(n);
        }
        y->Reshape({flat_from, flat_to});
    }
    else{
        runtime_assert(x->Dim(0) == y->Dim(0),
                       "[Flatten Op] x->Dim(0) == y->Dim(0).");
    }
}

template <class DT, class DC>
void FlattenOp<DT, DC>::Compute(){}

REGIST_OPERATOR_CPU(Flatten_float, FlattenOp<float, CPUContext>)
REGIST_OPERATOR_CPU(Flatten_double, FlattenOp<double, CPUContext>)

template <class DT, class DC>
FlattenGradientOp<DT, DC>::FlattenGradientOp(
                                                        OperatorIO &opio,
                                                        ItemHolder *ih
                                                        ) : Operator<DC>(opio, ih){
    runtime_assert(this->inputs.size() == 2,
                   "[Flatten Gradient Op] inputs.size() == 2.");
    runtime_assert(this->outputs.size() == 1,
                   "[Flatten Gradient Op] outputs.size() == 1.");
    
    const auto x = this->inputs[InputSchema::x];
    const auto dy = this->inputs[InputSchema::dy];
    auto dx = this->outputs[OutputSchema::dx];
    if(dx->IsEmpty() &&
       !x->IsEmpty()
       ){
        *dx = *dy;
        dx->Reshape(*x);
    }
    else{
        runtime_assert(dx->CompareSizeWith(*x),
                       "[Flatten Gradient Op] dx->CompareSizeWith(x).");
    }
}

template <class DT, class DC>
void FlattenGradientOp<DT, DC>::Compute(){}

REGIST_OPERATOR_CPU(Flatten_float_Gradient, FlattenGradientOp<float, CPUContext>)
REGIST_OPERATOR_CPU(Flatten_double_Gradient, FlattenGradientOp<double, CPUContext>)

struct FlattenGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(Flatten, FlattenGradientIO);

} /* namespace mlfe */
