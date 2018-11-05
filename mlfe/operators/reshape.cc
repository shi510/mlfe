#include "reshape.h"
#include "../core/op_dep.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(Reshape)
    .Input("X", "float32")
    .Output("Y", "float32")
    .Attr("shape", "int32s")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto y = odc->Output(0);
        auto shape = odc->GetAttr<std::vector<type::int32::T>>("shape");
        y.Reshape(shape, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(Reshape)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), type::float32());
    })
    .Finish();

class ReshapeGradient : public GradientHelper{
public:
    ReshapeGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        gpair[x] = functional::reshape(dy, x.Shape());

        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

} // end namespace functional
} // end namespace mlfe
