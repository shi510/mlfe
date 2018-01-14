#include "softmax_xent_with_label.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

template <class DT, class DC>
SoftmaxCrossEntropyWithLabelOp<DT, DC>
::SoftmaxCrossEntropyWithLabelOp(
                                 OperatorIO &opio,
                                 ItemHolder *ih
                                 ) : Operator<DC>(opio, ih){
    runtime_assert(inputs.size() == 2,
                   "[Softmax Cross Entropy With Label Op] inputs.size() == 2");
    runtime_assert(outputs.size() == 2,
                   "[Softmax Cross Entropy With Label Op] outputs.size() == 2");
    
    const auto x = inputs[InputSchema::x];
    const auto label = inputs[InputSchema::label];
    auto prob = outputs[OutputSchema::prob];
    auto loss = outputs[OutputSchema::loss];
    
    if(prob->IsEmpty() && loss->IsEmpty()){
        prob->Resize<DT>(*x);
        loss->Resize<DT>({1});
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
    
    sum_multiplier.Resize<DT, DC>({prob->Dim(1)});
    sum_multiplier.SetByConst<DT>((DT(1)));
    rows_max.Resize<DT, DC>({x->Dim(0)});
    scaler.Resize<DT, DC>({x->Dim(0)});
    
    /*
     * batch size.
     */
    m = x->Dim(0);
    /*
     * output size.
     */
    n = x->Dim(1);
}

template <class DT, class DC>
void SoftmaxCrossEntropyWithLabelOp<DT, DC>::Compute(){
    const auto x = inputs[InputSchema::x];
    const auto label = inputs[InputSchema::label];
    auto prob = outputs[OutputSchema::prob];
    auto loss = outputs[OutputSchema::loss];
    
    math::rowwise_max<DT, DC>(
                                         m, n,
                                         x->GetPtrConst<DT>(),
                                         rows_max.GetPtrMutable<DT>()
                                         );
    
    math::scal<DT, DC>(
                                  m * n, DT(1),
                                  x->GetPtrConst<DT>(),
                                  prob->GetPtrMutable<DT>()
                                  );
    
    math::gemm<DT, DC>(false, false,
                                  m, n, 1,
                                  DT(-1), rows_max.GetPtrConst<DT>(), 1,
                                  sum_multiplier.GetPtrConst<DT>(), n,
                                  DT(1), prob->GetPtrMutable<DT>(), n, nullptr);
    
    math::exp<DT, DC>(
                                 prob->Size(),
                                 prob->GetPtrConst<DT>(),
                                 prob->GetPtrMutable<DT>()
                                 );
    
    math::gemv<DT, DC>(false,
                                  m, n,
                                  DT(1), prob->GetPtrConst<DT>(), n,
                                  sum_multiplier.GetPtrConst<DT>(),
                                  DT(0), scaler.GetPtrMutable<DT>(), 1, nullptr);
    
    math::rowwise_normalize<DT, DC>(m, n,
                                               scaler.GetPtrConst<DT>(),
                                               prob->GetPtrMutable<DT>()
                                               );
    
    math::cross_entropy<DT, DC>(m, n,
                                           prob->GetPtrConst<DT>(),
                                           label->GetPtrConst<DT>(),
                                           rows_max.GetPtrMutable<DT>()
                                           );
    
    math::sum<DT,
    DC>(
                m,
                rows_max.GetPtrConst<DT>(),
                loss->GetPtrMutable<DT>()
                );
    
    math::scal<DT,
    DC>(
                1,
                static_cast<DT>(1) / static_cast<DT>(m),
                loss->GetPtrConst<DT>(),
                loss->GetPtrMutable<DT>()
                );
}

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel_float,
                    SoftmaxCrossEntropyWithLabelOp<float, CPUContext>)

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel_double,
                    SoftmaxCrossEntropyWithLabelOp<double, CPUContext>)

    template <class DT, class DC>
SoftmaxCrossEntropyWithLabelGradientOp<DT, DC>
::SoftmaxCrossEntropyWithLabelGradientOp(
                                         OperatorIO &opio,
                                         ItemHolder *ih
                                         ) : Operator<DC>(opio, ih) {
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
        dx->Resize<DT>(*x);
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

template <class DT, class DC>
void SoftmaxCrossEntropyWithLabelGradientOp<DT, DC>::Compute(){
    const auto prob = inputs[InputSchema::prob];
    const auto label = inputs[InputSchema::label];
    const auto loss = inputs[InputSchema::loss];
    auto dx = outputs[OutputSchema::dx];
    
    math::cross_entropy_gradients<DT, DC>(m, n,
                                                     prob->GetPtrConst<DT>(),
                                                     label->GetPtrConst<DT>(),
                                                     dx->GetPtrMutable<DT>()
                                                     );
    
    math::scal<DT, DC>(m * n,
                                  loss->GetPtrConst<DT>()[0] / static_cast<DT>(m),
                                  dx->GetPtrConst<DT>(),
                                  dx->GetPtrMutable<DT>()
                                  );
}

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel_float_Gradient, 
                    SoftmaxCrossEntropyWithLabelGradientOp<float, CPUContext>)

REGIST_OPERATOR_CPU(SoftmaxXentLossWithLabel_double_Gradient,
                    SoftmaxCrossEntropyWithLabelGradientOp<double, CPUContext>)

struct SoftmaxXentLossWithLabelGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
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
