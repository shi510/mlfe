#include "activations.h"
#include "../core/op_dep.h"
#include "../core/op_algo.h"
#include "../core/tensor.h"
#include "../core/gradient_helper.h"

namespace mlfe{ namespace functional{

REGIST_OP(ReLU)
    .Input("X", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto y = odc->Output(0);
        if (x.Name() != y.Name()){
            y.Reshape(x.Shape(), x.Type());
        }
    })
    .Finish();

REGIST_OP_GRAD(ReLU)
    .Input("X", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), x.Type());
    })
    .Finish();

class ReLUGradient : public GradientHelper{
public:
    ReLUGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        Tensor dx = create_variable(x.Shape());
        OpAlgoContext cxt("ReLUGradient");
        dx.add_child(x);
        dx.add_child(y);
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, cxt);
        gpair[x] = dx;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(ReLU, ReLUGradient)

REGIST_OP(Sigmoid)
    .Input("X", "float32")
    .Output("Y", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto y = odc->Output(0);
        if(x.Name() != y.Name()){
            y.Reshape(x.Shape(), x.Type());
        }
    })
    .Finish();

REGIST_OP_GRAD(Sigmoid)
    .Input("X", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("dX", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto x = odc->Input(0);
        auto dx = odc->Output(0);
        dx.Reshape(x.Shape(), x.Type());
    })
    .Finish();

class SigmoidGradient : public GradientHelper{
public:
    SigmoidGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    TensorUmap compute_gradient(Tensor y, 
                                Tensor dy
                               ) override{
        TensorUmap gpair;
        Tensor x = y.get_children()[0];
        Tensor dx = create_variable(x.Shape());
        OpAlgoContext cxt("SigmoidGradient");
        dx.add_child(x);
        dx.add_child(y);
        dx.add_child(dy);
        Tensor::AssignOpFunctor(dx, cxt);
        gpair[x] = dx;
        return gpair;
    }
};

REGIST_GRADIENT_HELPER(Sigmoid, SigmoidGradient)

Tensor relu(Tensor x){
    Tensor y = functional::create_variable(x.Shape());
    OpAlgoContext cxt("ReLU");
    y.add_child(x);
    Tensor::AssignOpFunctor(y, cxt);

    return y;
}

Tensor sigmoid(Tensor x){
    Tensor y = functional::create_variable(x.Shape());
    OpAlgoContext cxt("ReLU");
    y.add_child(x);
    Tensor::AssignOpFunctor(y, cxt);

    return y;
}

} // end namespace functional
} // end namespace mlfe
