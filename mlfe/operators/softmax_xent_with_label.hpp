#ifndef __SOFTMAX_XENT_WITH_LABEL_HPP__
#define __SOFTMAX_XENT_WITH_LABEL_HPP__

#include <numeric>
#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{
    
template <class DataType, class DeviceContext>
class SoftmaxCrossEntropyWithLabel final : public Operator<DeviceContext>{
public:
    explicit SoftmaxCrossEntropyWithLabel(
                                          std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                                          std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs
                                          ) : Operator<DeviceContext>(inputs, outputs, ParamDef("", 0)) {
        runtime_assert(inputs.size() == 2, "Input size must be 2(x, label).");
        runtime_assert(outputs.size() == 3, "Output size must be 3(probability, loss, dx).");
        
        const auto x = this->Input(InputSchema::x);
        const auto label = this->Input(InputSchema::label);
        auto prob = this->Output(OutputSchema::prob);
        auto loss = this->Output(OutputSchema::loss);
        auto dx = this->Output(OutputSchema::dx);
        
        runtime_assert(x->CompareSizeWith(label) , "x's size must be same with label.");
        runtime_assert(x->CompareSizeWith(prob) , "x's size must be same with prob.");
        runtime_assert(x->CompareSizeWith(dx) , "x's size must be same with dx.");
        runtime_assert(x->Dims() == 2, "x's dim size must be 2.");
        runtime_assert(prob->Dims() == 2, "probability's dim size must be 2.");
        runtime_assert(loss->Size() == 1, "loss's size must be 1.");
        for(int i = 0; i < label->Dim(0); ++i){
            std::vector<DataType> v(
                                    label->template GetPtrMutable<DataType>() + label->Dim(1) * (i),
                                    label->template GetPtrMutable<DataType>() + label->Dim(1) * (i + 1)
                                    );
            
            runtime_assert(static_cast<int>(std::accumulate(v.begin(), v.end(), DataType(0))) == 1, "label sum at each batch must be 1.");
            
        }
        
        sum_multiplier.template Reshape<DataType, DeviceContext>({prob->Dim(1)});
        sum_multiplier.template SetByConst<DataType>((DataType(1)));
        rows_max.template Reshape<DataType, DeviceContext>({x->Dim(0)});
        scaler.template Reshape<DataType, DeviceContext>({x->Dim(0)});
        
        /*
         * batch size.
         */
        m = x->Dim(0);
        /*
         * output size.
         */
        n = x->Dim(1);
    }
    
    ~SoftmaxCrossEntropyWithLabel() override {}
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        const auto label = this->Input(InputSchema::label);
        auto prob = this->Output(OutputSchema::prob);
        auto loss = this->Output(OutputSchema::loss);
        
        math::rowwise_max<DataType, DeviceContext>(
                                                   m, n,
                                                   x->template GetPtrConst<DataType>(),
                                                   rows_max.template GetPtrMutable<DataType>()
                                                   );
        
        math::scal<DataType, DeviceContext>(
                                         m * n, DataType(1),
                                         x->template GetPtrConst<DataType>(),
                                         prob->template GetPtrMutable<DataType>()
                                         );
        
        math::gemm<DataType, DeviceContext>(false, false,
                                            m, n, 1,
                                            DataType(-1), rows_max.template GetPtrConst<DataType>(), 1,
                                            sum_multiplier.template GetPtrConst<DataType>(), n,
                                            DataType(1), prob->template GetPtrMutable<DataType>(), n, nullptr);
        
        math::exp<DataType, DeviceContext>(
                                           prob->Size(),
                                           prob->template GetPtrConst<DataType>(),
                                           prob->template GetPtrMutable<DataType>()
                                           );
        
        math::gemv<DataType, DeviceContext>(false,
                                            m, n,
                                            DataType(1), prob->template GetPtrConst<DataType>(), n,
                                            sum_multiplier.template GetPtrConst<DataType>(),
                                            DataType(0), scaler.template GetPtrMutable<DataType>(), 1, nullptr);
        
        math::rowwise_normalize<DataType, DeviceContext>(m, n,
                                                         scaler.template GetPtrConst<DataType>(),
                                                         prob->template GetPtrMutable<DataType>()
                                                         );
        
        math::cross_entropy<DataType, DeviceContext>(m, n,
                                                     prob->template GetPtrConst<DataType>(),
                                                     label->template GetPtrConst<DataType>(),
                                                     loss->template GetPtrMutable<DataType>()
                                                     );
        
    }
    
    void ComputeGradients() override {
        const auto label = this->Input(InputSchema::label);
        const auto prob = this->Output(OutputSchema::prob);
        const auto loss = this->Output(OutputSchema::loss);
        auto dx = this->Output(OutputSchema::dx);
        
        math::cross_entropy_gradients<DataType, DeviceContext>(m, n,
                                                               prob->template GetPtrConst<DataType>(),
                                                               label->template GetPtrConst<DataType>(),
                                                               dx->template GetPtrMutable<DataType>()
                                                               );
        
        math::scal<DataType, DeviceContext>(m * n,
                                            loss->template GetPtrConst<DataType>()[0] / static_cast<DataType>(m),
                                            dx->template GetPtrConst<DataType>(),
                                            dx->template GetPtrMutable<DataType>()
                                            );
    }
    
private:
    enum InputSchema{ x, label };
    enum OutputSchema{ prob, loss, dx };
    TensorBlob<DeviceContext> sum_multiplier;
    TensorBlob<DeviceContext> rows_max;
    TensorBlob<DeviceContext> scaler;
    int m;
    int n;
};

} /* namespace mlfe */
#endif /* __SOFTMAX_XENT_WITH_LABEL_HPP__ */
