#ifndef __FULLY_CONNECTED_OP_HPP__
#define __FULLY_CONNECTED_OP_HPP__

#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{
    
template <class DataType, class DeviceContext>
class FullyConnectedOp final : public Operator<DeviceContext>{
public:
    explicit FullyConnectedOp(
                              std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> inputs,
                              std::vector<std::shared_ptr<TensorBlob<DeviceContext>>> outputs
                              ) : Operator<DeviceContext>(inputs, outputs, ParamDef("", 0)) {
        runtime_assert(inputs.size() == 4, "Input size must be 4(x, w, b, dy).");
        runtime_assert(outputs.size() == 4, "Output size must be 4(y, dw, db, dx).");
        
        const auto x = this->Input(InputSchema::x);
        const auto w = this->Input(InputSchema::w);
        const auto b = this->Input(InputSchema::b);
        const auto dy = this->Input(InputSchema::dy);
        auto y = this->Output(OutputSchema::y);
        auto dw = this->Output(OutputSchema::dw);
        auto db = this->Output(OutputSchema::db);
        auto dx = this->Output(OutputSchema::dx);
        
        runtime_assert(x->Dims() == 2, "x's dim size must be 2.");
        runtime_assert(x->Dim(0) == y->Dim(0), "x's dim(0) must be same with y's dim(0).");
        runtime_assert(x->Dim(1) == w->Dim(1), "x's dim(1) must be same with w's dim(1).");
        runtime_assert(y->Dim(1) == w->Dim(0), "y's dim(1) must be same with w's dim(0).");
        runtime_assert(dw->CompareSizeWith(w) , "dw's size must be same with w.");
        runtime_assert(db->CompareSizeWith(b) , "db's size must be same with b.");
        runtime_assert(dx->CompareSizeWith(x) , "dx's size must be same with x.");
        
        bias_multiplier.template Reshape<DataType, DeviceContext>({x->Dim(0)});
        bias_multiplier.template SetByConst<DataType>(DataType(1));
        
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
    
    ~FullyConnectedOp() override {}
    
    void Compute() override {
        const auto x = this->Input(InputSchema::x);
        const auto w = this->Input(InputSchema::w);
        const auto b = this->Input(InputSchema::b);
        auto y = this->Output(OutputSchema::y);
        /*
         * Forward computation.
         * x(batch_size x input_size) * w(output_size x input_size)^T
         *  = y(batch_size x output_size)
         */
        math::gemm<DataType, DeviceContext>(
                                         false, true,
                                         m, n, k,
                                         DataType(1), x->template GetPtrConst<DataType>(), k,
                                         w->template GetPtrConst<DataType>(), k,
                                         DataType(0), y->template GetPtrMutable<DataType>(), n, nullptr
                                         );
        
        /*
         * Add the bias term.
         * y = y + b;
         */
        
        math::gemm<DataType, DeviceContext>(
                                         false, false,
                                         m, n, 1,
                                         DataType(1), bias_multiplier.template GetPtrConst<DataType>(), 1
                                         , b->template GetPtrConst<DataType>(), n,
                                         DataType(1), y->template GetPtrMutable<DataType>(), n, nullptr
                                         );
    }
    
    void ComputeGradients() override {
        const auto x = this->Input(InputSchema::x);
        const auto w = this->Input(InputSchema::w);
        const auto dy = this->Input(InputSchema::dy);
        auto dw = this->Output(OutputSchema::dw);
        auto db = this->Output(OutputSchema::db);
        auto dx = this->Output(OutputSchema::dx);
        /*
         * db = dy.
         */
        math::gemv<DataType, DeviceContext>(true, m, n, DataType(1),
                                      dy->template GetPtrConst<DataType>(), n,
                                      bias_multiplier.template GetPtrConst<DataType>(), DataType(0),
                                      db->template GetPtrMutable<DataType>(), n, nullptr);
        
        /*
         * Calculate gradients of weights.
         * dy(batch_size x output_size)^T * x(batch_size x input_size)
         *  = dw(output_size x input_size)
         */
        math::gemm<DataType, DeviceContext>(true, false,
                                      n, k, m,
                                      DataType(1), dy->template GetPtrConst<DataType>(), n,
                                      x->template GetPtrConst<DataType>(), k,
                                      DataType(0), dw->template GetPtrMutable<DataType>(), k, nullptr);
        
        /*
         * Calculate loss to propagate through bottom.
         * dy(batch_size x output_size) * w(output_size x input_size)
         *  = dx(batch_size x input_size)
         */
        math::gemm<DataType, DeviceContext>(
                                      false, false,
                                      m, k, n,
                                      DataType(1), dy->template GetPtrConst<DataType>(), n,
                                      w->template GetPtrConst<DataType>(), k,
                                      DataType(0), dx->template GetPtrMutable<DataType>(), k, nullptr);
        
        math::scal<DataType, DeviceContext>(
                                      db->Size(),
                                      DataType(1) / static_cast<DataType>(x->Dim(0)),
                                      db->template GetPtrConst<DataType>(),
                                      db->template GetPtrMutable<DataType>()
                                      );
        
        math::scal<DataType, DeviceContext>(
                                      dw->Size(),
                                      DataType(1) / static_cast<DataType>(x->Dim(0)),
                                      dw->template GetPtrConst<DataType>(),
                                      dw->template GetPtrMutable<DataType>()
                                      );
    }
    
private:
    enum InputSchema{ x, w, b, dy};
    enum OutputSchema{ y, dw, db, dx};
    TensorBlob<DeviceContext> bias_multiplier;
    int m;
    int n;
    int k;
};

} /* namespace mlfe */
#endif /* __FULLY_CONNECTED_OP_HPP__ */
