#include "fully_connected.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe{

template <>
FullyConnectedOp<float, CPUContext>::FullyConnectedOp(
                                                      OperatorIO &opio,
                                                      ItemHolder *ih
                                                      ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 3,
                   "[Fully Connected Op] inputs.size() == 3.");
    runtime_assert(outputs.size() == 1,
                   "[Fully Connected Op] outputs.size() == 1.");
    
    const auto x = inputs[InputSchema::x];
    const auto w = inputs[InputSchema::w];
    const auto b = inputs[InputSchema::b];
    auto y = outputs[OutputSchema::y];
    int units;
    
    if(opio.param.HasParam("Units") &&
       w->IsEmpty() &&
       b->IsEmpty() &&
       y->IsEmpty() &&
       !x->IsEmpty() &&
       x->Dims() == 2){
        units = opio.param.GetParam<int>("Units");
        w->Resize<float>({units, x->Dim(1)});
        b->Resize<float>({units});
        y->Resize<float>({x->Dim(0), units});
    }
    else{
        runtime_assert(x->Dims() == 2,
                       "[Fully Connected Op] x->Dims() == 2.");
        runtime_assert(x->Dim(0) == y->Dim(0),
                       "[Fully Connected Op] x->Dim(0) == y->Dim(0).");
        runtime_assert(x->Dim(1) == w->Dim(1),
                       "[Fully Connected Op] x->Dim(1) == w->Dim(1).");
        runtime_assert(y->Dim(1) == w->Dim(0),
                       "[Fully Connected Op] y->Dim(1) == w->Dim(0).");
    }
    
    bias_multiplier.Resize<float, CPUContext>({x->Dim(0)});
    bias_multiplier.SetByConst<float>(float(1));
    
    /*
     * batch size.
     */
    m = x->Dim(0);
    /*
     * output size.
     */
    n = w->Dim(0);
    /*
     * total input's element size.
     */
    k = w->Dim(1);
}

template <>
void FullyConnectedOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    const auto w = inputs[InputSchema::w];
    const auto b = inputs[InputSchema::b];
    auto y = outputs[OutputSchema::y];
    /*
     * Forward computation.
     * x(batch_size x input_size) * w(output_size x input_size)^T
     *  = y(batch_size x output_size)
     */
    math::gemm<float, CPUContext>(
                                  false, true,
                                  m, n, k,
                                  float(1), x->GetPtrConst<float>(), k,
                                  w->GetPtrConst<float>(), k,
                                  float(0), y->GetPtrMutable<float>(), n, nullptr
                                  );
    
    /*
     * Add the bias term.
     * y = y + b;
     */
    
    math::gemm<float, CPUContext>(
                                  false, false,
                                  m, n, 1,
                                  float(1), bias_multiplier.GetPtrConst<float>(), 1
                                  , b->GetPtrConst<float>(), n,
                                  float(1), y->GetPtrMutable<float>(), n, nullptr
                                  );
}

REGIST_OPERATOR_CPU(FC, FullyConnectedOp<float, CPUContext>)

template <>
FullyConnectedGradientOp<float, CPUContext>::FullyConnectedGradientOp(
                                                                      OperatorIO &opio,
                                                                      ItemHolder *ih
                                                                      ) : Operator<CPUContext>(opio, ih){
    runtime_assert(inputs.size() == 3,
                   "[Fully Connected Gradient Op] inputs.size() == 3.");
    runtime_assert(outputs.size() == 3,
                   "[Fully Connected Gradient Op] outputs.size() == 3.");
    
    const auto x = inputs[InputSchema::x];
    const auto w = inputs[InputSchema::w];
    const auto dy = inputs[InputSchema::dy];
    auto dw = outputs[OutputSchema::dw];
    auto db = outputs[OutputSchema::db];
    auto dx = outputs[OutputSchema::dx];
    int units;
    if(opio.param.HasParam("Units") &&
       dw->IsEmpty() &&
       db->IsEmpty() &&
       dx->IsEmpty() &&
       !dy->IsEmpty() &&
       !x->IsEmpty() &&
       x->Dims() == 2
       ){
        units = opio.param.GetParam<int>("Units");
        dw->Resize<float>(*w);
        db->Resize<float>({units});
        dx->Resize<float>(*x);
    }
    else{
        runtime_assert(x->Dims() == 2,
                       "[Fully Connected Gradient Op]x->Dims() == 2.");
        runtime_assert(x->Dim(1) == w->Dim(1),
                       "[Fully Connected Gradient Op] x->Dim(1) == w->Dim(1).");
        runtime_assert(dw->CompareSizeWith(*w),
                       "[Fully Connected Gradient Op] dw->CompareSizeWith(w).");
        runtime_assert(dx->CompareSizeWith(*x),
                       "[Fully Connected Gradient Op] dx->CompareSizeWith(x).");
    }
    
    bias_multiplier.Resize<float, CPUContext>({x->Dim(0)});
    bias_multiplier.SetByConst<float>(float(1));
    
    /*
     * batch size.
     */
    m = x->Dim(0);
    /*
     * output size.
     */
    n = w->Dim(0);
    /*
     * total input's element size.
     */
    k = w->Dim(1);
}

template <>
void FullyConnectedGradientOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    const auto w = inputs[InputSchema::w];
    const auto dy = inputs[InputSchema::dy];
    auto dw = outputs[OutputSchema::dw];
    auto db = outputs[OutputSchema::db];
    auto dx = outputs[OutputSchema::dx];
    /*
     * db = dy.
     */
    math::gemv<float, CPUContext>(true, m, n, float(1),
                                  dy->GetPtrConst<float>(), n,
                                  bias_multiplier.GetPtrConst<float>(), float(0),
                                  db->GetPtrMutable<float>(), n, nullptr);
    
    /*
     * Calculate gradients of weights.
     * dy(batch_size x output_size)^T * x(batch_size x input_size)
     *  = dw(output_size x input_size)
     */
    math::gemm<float, CPUContext>(true, false,
                                  n, k, m,
                                  float(1), dy->GetPtrConst<float>(), n,
                                  x->GetPtrConst<float>(), k,
                                  float(0), dw->GetPtrMutable<float>(), k, nullptr);
    
    /*
     * Calculate loss to propagate through bottom.
     * dy(batch_size x output_size) * w(output_size x input_size)
     *  = dx(batch_size x input_size)
     */
    math::gemm<float, CPUContext>(
                                  false, false,
                                  m, k, n,
                                  float(1), dy->GetPtrConst<float>(), n,
                                  w->GetPtrConst<float>(), k,
                                  float(0), dx->GetPtrMutable<float>(), k, nullptr);
    
    math::scal<float, CPUContext>(
                                  db->Size(),
                                  float(1) / static_cast<float>(x->Dim(0)),
                                  db->GetPtrConst<float>(),
                                  db->GetPtrMutable<float>()
                                  );
    
    math::scal<float, CPUContext>(
                                  dw->Size(),
                                  float(1) / static_cast<float>(x->Dim(0)),
                                  dw->GetPtrConst<float>(),
                                  dw->GetPtrMutable<float>()
                                  );
}

REGIST_OPERATOR_CPU(FC_Gradient, FullyConnectedGradientOp<float, CPUContext>)

struct FCGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_Gradient";
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.inputs[1]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[1] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[2] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(FC, FCGradientIO);

} /* namespace mlfe */
