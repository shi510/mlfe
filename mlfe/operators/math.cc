#include "math.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "../operators/basic_arithmetics.h"
#include "../operators/initializer.h"

namespace mlfe{

REGIST_OP(SquaredDifference)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Output("Y", "float32")
    .Attr("output", "int32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto y = odc->Output(0);
        
        if(x1.dims() != x2.dims()){
            throw std::string("squared_difference op : ") +
                std::string("x1.shape() != x2.shape()");
        }
        for(int n = 0; n < x1.dims(); ++n){
            if(x1.shape()[n] != x2.shape()[n]){
                throw std::string("squared_difference op : ") +
                    std::string("x1.shape() != x2.shape()");
            }
        }
        y.reshape(x1.shape(), type::float32());
    })
    .Finish();

REGIST_OP_GRAD(SquaredDifference)
    .Input("X1", "float32")
    .Input("X2", "float32")
    .Input("dY", "float32")
    .Output("dX1", "float32")
    .Output("dX2", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x1 = odc->Input(0);
        auto x2 = odc->Input(1);
        auto dy = odc->Input(2);
        auto dx1 = odc->Output(0);
        auto dx2 = odc->Output(1);
        dx1.reshape(x1.shape(), type::float32());
        dx2.reshape(x2.shape(), type::float32());
    })
    .Finish();

class SquaredDifferenceGradient : public GradientHelper{
public:
    SquaredDifferenceGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        namespace fn = functional;
        VecTensor in_grads;
        auto x1 = y.get_children()[0];
        auto x2 = y.get_children()[1];
        auto two = functional::constant(2, x1.shape());
        Tensor dx1 = fn::mul(dy, fn::mul(two, fn::sub(x1, x2)));
        Tensor dx2 = fn::mul(dy, fn::negative(fn::mul(two, fn::sub(x1, x2))));
        in_grads.push_back(dx1);
        in_grads.push_back(dx2);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(SquaredDifference, SquaredDifferenceGradient)

REGIST_OP(ReduceMean)
.Input("X", "float32")
.Output("Y", "float32")
.ShapeInference([](OpDesignContext * odc){
    auto x = odc->Input(0);
    auto y = odc->Output(0);
    y.reshape({ 1 }, type::float32());
})
.Finish();

REGIST_OP_GRAD(ReduceMean)
.Input("X", "float32")
.Input("dY", "float32")
.Output("dX", "float32")
.ShapeInference([](OpDesignContext * odc){
    auto x = odc->Input(0);
    auto dx = odc->Output(0);
    dx.reshape(x.shape(), type::float32());
})
.Finish();

class ReduceMeanGradient : public GradientHelper{
public:
    ReduceMeanGradient(const OpDesignContext *odc)
    : GradientHelper(odc){}
    
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor x = y.get_children()[0];
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("ReduceMeanGradient");
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(ReduceMean, ReduceMeanGradient)

namespace functional{

Tensor squared_difference(Tensor x1, Tensor x2){
    Tensor y = create_variable(x1.shape());
    OpAlgoContext ctx("SquaredDifference");
    y.add_child(x1);
    y.add_child(x2);
    Tensor::AssignOpFunctor(y, ctx);
    return y;
}

Tensor mean(Tensor x){
    Tensor y = create_variable({1});
    OpAlgoContext ctx("ReduceMean");
    
    y.add_child(x);
    Tensor::AssignOpFunctor(y, ctx);
    
    return y;
}

} // end namespace functional
} // end namespace mlfe
