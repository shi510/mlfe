#include "initializer.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/op_design.h"
#include "../core/gradient_helper.h"

namespace mlfe{

REGIST_OP(Constant)
    .Output("Y", "float32")
    .Attr("value", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto y = odc->Output(0);
        auto shape = odc->GetAttr<std::vector<int>>("shape");
        y.Reshape(shape);
    })
    .Finish();

REGIST_OP(Normal)
    .Output("Y", "float32")
    .Attr("std", "float32")
    .Attr("clip", "bool")
    .ShapeInference([](OpDesignContext * odc){
        auto y = odc->Output(0);
        auto shape = odc->GetAttr<std::vector<int>>("shape");
        y.Reshape(shape);
    })
    .Finish();

class ConstantGradient : public GradientHelper{
public:
    ConstantGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        gpair[y] = dy;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Constant, ConstantGradient)

class NormalGradient : public GradientHelper{
public:
    NormalGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        gpair[y] = dy;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Normal, NormalGradient)

namespace functional{

Tensor constant(type::float64::T val, std::vector<int> shape){
    Tensor y = create_variable(shape);
    OpAlgoContext ctx("Constant");
    ctx.add_attr({"value", static_cast<type::float32::T>(val)});
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

Tensor normal(type::float64::T std, std::vector<int> shape){
    Tensor y = create_variable(shape);
    OpAlgoContext ctx("Normal");
    ctx.add_attr({"std", static_cast<type::float32::T>(std)});
    ctx.add_attr({"clrip", static_cast<bool>(false)});
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

Tensor truncated_normal(type::float64::T std, std::vector<int> shape){
    Tensor y = create_variable(shape);
    OpAlgoContext ctx("Normal");
    ctx.add_attr({"std", static_cast<type::float32::T>(std)});
    ctx.add_attr({"clrip", static_cast<bool>(true)});
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
