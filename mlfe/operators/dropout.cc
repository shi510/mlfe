#include "dropout.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(Dropout)
    .Input("X", "float32")
    .Output("Y", "float32")
    .Output("Mask", "float32")
    .Attr("dropout_ratio", "float32")
    .Attr("is_training_step", "bool")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto y = odc->Output(0);
        auto mask = odc->Output(1);
        y.Reshape(x.Shape(), type::float32());
        mask.Reshape(x.Shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(Dropout)
    .Input("X", "float32")
    .Input("Y", "float32")
    .Input("Mask", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .Attr("dropout_ratio", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class DropoutGradient : public GradientHelper{
public:
    DropoutGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = odc->Input(0);
        Tensor mask = odc->Output(1);
        Tensor dx;

        dep = OpDependency::Builder("DenseGradient")
            .Input(x).Input(y).Input(mask).Input(dy)
            .Output(dx)
            .Attr({"dropout_ratio", odc->GetAttr<float>("dropout_ratio")})
            .Finish();

        gpair[x] = dx;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Dropout, DropoutGradient)

Tensor Dropout(Tensor x, type::float64::T probability, bool is_training){
    Tensor y, dropout_mask;

    auto dep = OpDependency::Builder("Dropout")
        .Input(x)
        .Output(y)
        .Output(dropout_mask)
        .Attr({ "dropout_ratio", float(probability) })
        .Attr({ "is_training_step", is_training })
        .Finish();

    y = Tensor::DependencyAdder(dep);
    y.add_child(x);

    return y;
}

} // end namespace functional
} // end namespace mlfe
