#ifndef __FULLY_CONNECTED_OP_HPP__
#define __FULLY_CONNECTED_OP_HPP__

#include "operator.hpp"
#include "../core/tensor_blob.hpp"
#include "../core/param_def.hpp"
#include "../math/blas.hpp"
#include "../utils/assert.hpp"

namespace mlfe{
    
template <class DT, class DC>
class FullyConnectedOp final : public Operator<DC>{
public:
    explicit FullyConnectedOp(OperatorIO &opio, ItemHolder *ih) 
        : Operator<DC>(opio, ih) {
        runtime_assert(this->inputs.size() == 3,
            "[Fully Connected Op] inputs.size() == 3.");
        runtime_assert(this->outputs.size() == 1,
            "[Fully Connected Op] outputs.size() == 1.");

        const auto x = this->inputs[InputSchema::x];
        const auto w = this->inputs[InputSchema::w];
        const auto b = this->inputs[InputSchema::b];
        auto y = this->outputs[OutputSchema::y];
        int units;

        if (opio.param.HasParam("Units") &&
            w->IsEmpty() &&
            b->IsEmpty() &&
            y->IsEmpty() &&
            !x->IsEmpty() &&
            x->Dims() == 2) {
            units = opio.param.GetParam<int>("Units");
            w->template Resize<DT>({ units, x->Dim(1) });
            b->template Resize<DT>({ units });
            y->template Resize<DT>({ x->Dim(0), units });
        }
        else {
            runtime_assert(x->Dims() == 2,
                "[Fully Connected Op] x->Dims() == 2.");
            runtime_assert(x->Dim(0) == y->Dim(0),
                "[Fully Connected Op] x->Dim(0) == y->Dim(0).");
            runtime_assert(x->Dim(1) == w->Dim(1),
                "[Fully Connected Op] x->Dim(1) == w->Dim(1).");
            runtime_assert(y->Dim(1) == w->Dim(0),
                "[Fully Connected Op] y->Dim(1) == w->Dim(0).");
        }

        bias_multiplier.template Resize<DT, DC>({ x->Dim(0) });
        math::set<DT, DC>(
            bias_multiplier.Size(),
            static_cast<DT>(1),
            bias_multiplier.template GetPtrMutable<DT>()
            );

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
    
    void Compute() override {
        const auto x = this->inputs[InputSchema::x];
        const auto w = this->inputs[InputSchema::w];
        const auto b = this->inputs[InputSchema::b];
        auto y = this->outputs[OutputSchema::y];
        /*
        * Forward computation.
        * x(batch_size x input_size) * w(output_size x input_size)^T
        *  = y(batch_size x output_size)
        */
        math::gemm<DT, DC>(
            false, true,
            m, n, k,
            DT(1), x->template GetPtrConst<DT>(), k,
            w->template GetPtrConst<DT>(), k,
            DT(0), y->template GetPtrMutable<DT>(), n, nullptr
            );

        /*
        * Add the bias term.
        * y = y + b;
        */

        math::gemm<DT, DC>(
            false, false,
            m, n, 1,
            DT(1), bias_multiplier.template GetPtrConst<DT>(), 1
            , b->template GetPtrConst<DT>(), n,
            DT(1), y->template GetPtrMutable<DT>(), n, nullptr
            );
    }
    
private:
    enum InputSchema{x, w, b};
    enum OutputSchema{y};
    TensorBlob<DC> bias_multiplier;
    int m;
    int n;
    int k;
};
    
template <class DT, class DC>
class FullyConnectedGradientOp final : public Operator<DC>{
public:
    explicit FullyConnectedGradientOp(
        OperatorIO &opio, 
        ItemHolder *ih) : Operator<DC>(opio, ih) {
        runtime_assert(this->inputs.size() == 3,
            "[Fully Connected Gradient Op] inputs.size() == 3.");
        runtime_assert(this->outputs.size() == 3,
            "[Fully Connected Gradient Op] outputs.size() == 3.");

        const auto x = this->inputs[InputSchema::x];
        const auto w = this->inputs[InputSchema::w];
        const auto dy = this->inputs[InputSchema::dy];
        auto dw = this->outputs[OutputSchema::dw];
        auto db = this->outputs[OutputSchema::db];
        auto dx = this->outputs[OutputSchema::dx];
        int units;
        if (opio.param.HasParam("Units") &&
            dw->IsEmpty() &&
            db->IsEmpty() &&
            dx->IsEmpty() &&
            !dy->IsEmpty() &&
            !x->IsEmpty() &&
            x->Dims() == 2
            ) {
            units = opio.param.GetParam<int>("Units");
            dw->template Resize<DT>(*w);
            db->template Resize<DT>({ units });
            dx->template Resize<DT>(*x);
        }
        else {
            runtime_assert(x->Dims() == 2,
                "[Fully Connected Gradient Op]x->Dims() == 2.");
            runtime_assert(x->Dim(1) == w->Dim(1),
                "[Fully Connected Gradient Op] x->Dim(1) == w->Dim(1).");
            runtime_assert(dw->CompareSizeWith(*w),
                "[Fully Connected Gradient Op] dw->CompareSizeWith(w).");
            runtime_assert(dx->CompareSizeWith(*x),
                "[Fully Connected Gradient Op] dx->CompareSizeWith(x).");
        }

        bias_multiplier.template Resize<DT, DC>({ x->Dim(0) });
        math::set<DT, DC>(
            bias_multiplier.Size(),
            static_cast<DT>(1),
            bias_multiplier.template GetPtrMutable<DT>()
            );

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
    
    void Compute() override {
        const auto x = this->inputs[InputSchema::x];
        const auto w = this->inputs[InputSchema::w];
        const auto dy = this->inputs[InputSchema::dy];
        auto dw = this->outputs[OutputSchema::dw];
        auto db = this->outputs[OutputSchema::db];
        auto dx = this->outputs[OutputSchema::dx];
        /*
        * db = dy.
        */
        math::gemv<DT, DC>(true, m, n, DT(1),
            dy->template GetPtrConst<DT>(), n,
            bias_multiplier.template GetPtrConst<DT>(), DT(0),
            db->template GetPtrMutable<DT>(), n, nullptr);

        /*
        * Calculate gradients of weights.
        * dy(batch_size x output_size)^T * x(batch_size x input_size)
        *  = dw(output_size x input_size)
        */
        math::gemm<DT, DC>(true, false,
            n, k, m,
            DT(1), dy->template GetPtrConst<DT>(), n,
            x->template GetPtrConst<DT>(), k,
            DT(0), dw->template GetPtrMutable<DT>(), k, nullptr);

        /*
        * Calculate loss to propagate through bottom.
        * dy(batch_size x output_size) * w(output_size x input_size)
        *  = dx(batch_size x input_size)
        */
        math::gemm<DT, DC>(
            false, false,
            m, k, n,
            DT(1), dy->template GetPtrConst<DT>(), n,
            w->template GetPtrConst<DT>(), k,
            DT(0), dx->template GetPtrMutable<DT>(), k, nullptr);

        math::scal<DT, DC>(
            db->Size(),
            DT(1) / static_cast<DT>(x->Dim(0)),
            db->template GetPtrConst<DT>(),
            db->template GetPtrMutable<DT>()
            );

        math::scal<DT, DC>(
            dw->Size(),
            DT(1) / static_cast<DT>(x->Dim(0)),
            dw->template GetPtrConst<DT>(),
            dw->template GetPtrMutable<DT>()
            );
    }
    
private:
    enum InputSchema{x, w, dy};
    enum OutputSchema{dw, db, dx};
    TensorBlob<DC> bias_multiplier;
    int m;
    int n;
    int k;
};

} /* namespace mlfe */
#endif /* __FULLY_CONNECTED_OP_HPP__ */
