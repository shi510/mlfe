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
        auto x = odc->Input("X");
        auto y = odc->Output("Y");
        auto mask = odc->Output("Mask");
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
        auto x = odc->Input("X");
        auto dx = odc->Output("dX");
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class DropoutGradient : public GradientHelper{
public:
    DropoutGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    GradientHelper::HelperOut Get(Tensor dy) override{
        Tensor x = odc->Input("X");
        Tensor y = odc->Output("Y");
        Tensor mask = odc->Output("Mask");
        Tensor dx;
        GradientHelper::GradientPairs pairs;

        auto dep = OpDependency::Builder("DropoutGradient")
            .Input(std::make_tuple("X", x ))
            .Input(std::make_tuple("Y", y))
            .Input(std::make_tuple("Mask", mask))
            .Input(std::make_tuple(Gradient("Y"), dy))
            .Output(std::make_tuple(Gradient("X"), dx))
            .Attr({ "dropout_ratio", odc->GetAttr<float>("dropout_ratio") })
            .Finish();

        dx = Tensor::DependencyAdder(dep);

        return std::make_tuple(dx, pairs);
    }
};

REGIST_GRADIENT_HELPER(Dropout, DropoutGradient)

Tensor Dropout(Tensor x, type::float64::T probability, bool is_training){
    Tensor y, dropout_mask;

    auto dep = OpDependency::Builder("Dropout")
        .Input(std::make_tuple("X", x))
        .Output(std::make_tuple("Y", y))
        .Output(std::make_tuple("Mask", dropout_mask))
        .Attr({ "dropout_ratio", float(probability) })
        .Attr({ "is_training_step", is_training })
        .Finish();

    y = Tensor::DependencyAdder(dep);

    return y;
}

} // end namespace functional
} // end namespace mlfe
