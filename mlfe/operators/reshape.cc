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
        y.reshape(shape, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(Reshape)
    .Input("X", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.reshape(x.shape(), type::float32());
    })
    .Finish();

class ReshapeGradient : public GradientHelper{
public:
    ReshapeGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = functional::reshape(dy, x.shape());
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Reshape, ReshapeGradient)

} // end namespace functional
} // end namespace mlfe
