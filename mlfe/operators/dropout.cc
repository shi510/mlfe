#include "dropout.h"
#include "../core/op_algo.h"
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
        Tensor x = y.get_children()[0];
        Tensor prob = y.get_children()[1];
        Tensor mask = y.get_children()[2];
        Tensor dx = functional::create_variable(x.Shape());
        OpAlgoContext ctx("DenseGradient");
        dx.add_child(x);
        dx.add_child(prob);
        dx.add_child(mask);
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx);

        gpair[x] = dx;

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Dropout, DropoutGradient)

Tensor Dropout(Tensor x, Tensor prob){
    Tensor y = functional::create_variable(x.Shape());
    Tensor dropout_mask = functional::create_variable(x.Shape());
    OpAlgoContext ctx("Dropout");
    y.add_child(x);
    y.add_child(prob);
    y.add_child(dropout_mask);
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
