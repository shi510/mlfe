#include "softmax_xent_with_label.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <>
SoftmaxCrossEntropyWithLabelOp<float, CPUContext>
::SoftmaxCrossEntropyWithLabelOp(
                                 OperatorIO &opio,
                                 ItemHolder *ih
                                 ) : Operator<CPUContext>(opio, ih){
    runtime_assert(inputs.size() == 2,
                   "[Softmax Cross Entropy With Label Op] inputs.size() == 2");
    runtime_assert(outputs.size() == 2,
                   "[Softmax Cross Entropy With Label Op] outputs.size() == 2");
    
    const auto x = inputs[InputSchema::x];
    const auto label = inputs[InputSchema::label];
    auto prob = outputs[OutputSchema::prob];
    auto loss = outputs[OutputSchema::loss];
    
    if(prob->IsEmpty() && loss->IsEmpty()){
        prob->Resize<float>(*x);
        loss->Resize<float>({1});
    }
    else{
        runtime_assert(x->CompareSizeWith(*label),
                       "[Softmax Cross Entropy With Label Op] x->CompareSizeWith(label).");
        runtime_assert(x->CompareSizeWith(*prob),
                       "[Softmax Cross Entropy With Label Op]x->CompareSizeWith(prob).");
        runtime_assert(x->Dims() == 2,
                       "[Softmax Cross Entropy With Label Op] x->Dims() == 2.");
        runtime_assert(prob->Dims() == 2,
                       "[Softmax Cross Entropy With Label Op] prob->Dims() == 2");
        runtime_assert(loss->Size() == 1,
                       "[Softmax Cross Entropy With Label Op] loss->Size() == 1");
    }
    
    sum_multiplier.Resize<float, CPUContext>({prob->Dim(1)});
    sum_multiplier.SetByConst<float>((float(1)));
    rows_max.Resize<float, CPUContext>({x->Dim(0)});
    scaler.Resize<float, CPUContext>({x->Dim(0)});
    
    /*
     * batch size.
     */
    m = x->Dim(0);
    /*
     * output size.
     */
    n = x->Dim(1);
}

template <>
void SoftmaxCrossEntropyWithLabelOp<float, CPUContext>::Compute(){
    const auto x = inputs[InputSchema::x];
    const auto label = inputs[InputSchema::label];
    auto prob = outputs[OutputSchema::prob];
    auto loss = outputs[OutputSchema::loss];
    
    math::rowwise_max<float, CPUContext>(
                                         m, n,
                                         x->GetPtrConst<float>(),
                                         rows_max.GetPtrMutable<float>()
                                         );
    
    math::scal<float, CPUContext>(
                                  m * n, float(1),
                                  x->GetPtrConst<float>(),
                                  prob->GetPtrMutable<float>()
                                  );
    
    math::gemm<float, CPUContext>(false, false,
                                  m, n, 1,
                                  float(-1), rows_max.GetPtrConst<float>(), 1,
                                  sum_multiplier.GetPtrConst<float>(), n,
                                  float(1), prob->GetPtrMutable<float>(), n, nullptr);
    
    math::exp<float, CPUContext>(
                                 prob->Size(),
                                 prob->GetPtrConst<float>(),
                                 prob->GetPtrMutable<float>()
                                 );
    
    math::gemv<float, CPUContext>(false,
                                  m, n,
                                  float(1), prob->GetPtrConst<float>(), n,
                                  sum_multiplier.GetPtrConst<float>(),
                                  float(0), scaler.GetPtrMutable<float>(), 1, nullptr);
    
    math::rowwise_normalize<float, CPUContext>(m, n,
                                               scaler.GetPtrConst<float>(),
                                               prob->GetPtrMutable<float>()
                                               );
    
    math::cross_entropy<float, CPUContext>(m, n,
                                           prob->GetPtrConst<float>(),
                                           label->GetPtrConst<float>(),
                                           rows_max.GetPtrMutable<float>()
                                           );
    
    math::sum<float,
    CPUContext>(
                m,
                rows_max.GetPtrConst<float>(),
                loss->GetPtrMutable<float>()
                );
    
    math::scal<float,
    CPUContext>(
                1,
                static_cast<float>(1) / static_cast<float>(m),
                loss->GetPtrConst<float>(),
                loss->GetPtrMutable<float>()
                );
}

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel,
                    SoftmaxCrossEntropyWithLabelOp<float, CPUContext>)

template <>
SoftmaxCrossEntropyWithLabelGradientOp<float, CPUContext>
::SoftmaxCrossEntropyWithLabelGradientOp(
                                         OperatorIO &opio,
                                         ItemHolder *ih
                                         ) : Operator<CPUContext>(opio, ih) {
    runtime_assert(inputs.size() == 4,
                   "[Softmax Cross Entropy With Label Gradient Op] inputs.size() == 4");
    runtime_assert(outputs.size() == 1,
                   "[Softmax Cross Entropy With Label Gradient Op] outputs.size() == 1");
    
    const auto x = inputs[InputSchema::x];
    const auto label = inputs[InputSchema::label];
    const auto prob = inputs[InputSchema::prob];
    const auto loss = inputs[InputSchema::loss];
    auto dx = outputs[OutputSchema::dx];
    
    runtime_assert(prob->Dims() == 2,
                   "[Softmax Cross Entropy With Label Gradient Op] prob->Dims() == 2.");
    runtime_assert(loss->Size() == 1,
                   "[Softmax Cross Entropy With Label Gradient Op] loss->Size() == 1.");
    if(dx->IsEmpty()){
        dx->Resize<float>(*x);
    }
    else{
        runtime_assert(dx->CompareSizeWith(*x),
                       "[Softmax Cross Entropy With Label Gradient Op] dx->CompareSizeWith(x).");
        runtime_assert(prob->CompareSizeWith(*label),
                       "[Softmax Cross Entropy With Label Gradient Op] prob->CompareSizeWith(label).");
    }
    
    /*
     * batch size.
     */
    m = dx->Dim(0);
    /*
     * output size.
     */
    n = dx->Dim(1);
}

template <>
void SoftmaxCrossEntropyWithLabelGradientOp<float, CPUContext>::Compute(){
    const auto prob = inputs[InputSchema::prob];
    const auto label = inputs[InputSchema::label];
    const auto loss = inputs[InputSchema::loss];
    auto dx = outputs[OutputSchema::dx];
    
    math::cross_entropy_gradients<float, CPUContext>(m, n,
                                                     prob->GetPtrConst<float>(),
                                                     label->GetPtrConst<float>(),
                                                     dx->GetPtrMutable<float>()
                                                     );
    
    math::scal<float, CPUContext>(m * n,
                                  loss->GetPtrConst<float>()[0] / static_cast<float>(m),
                                  dx->GetPtrConst<float>(),
                                  dx->GetPtrMutable<float>()
                                  );
}

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel_Gradient, SoftmaxCrossEntropyWithLabelGradientOp<float, CPUContext>)

struct SoftmaxXentLossWithLabelGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_Gradient";
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.inputs[1]);
        opio_grad.inputs.push_back(opio.outputs[0]);
        opio_grad.inputs.push_back(opio.outputs[1]);
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(SoftmaxXentLossWithLabel, SoftmaxXentLossWithLabelGradientIO);

} /* namespace mlfe */
