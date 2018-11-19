#include "matmul.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "../utils/assert.h"

namespace mlfe{

REGIST_OP(MatMul)
    .Input("A", "float32")
    .Input("B", "float32")
    .Output("Y", "float32")
    .Attr("trans_a", "bool")
    .Attr("trans_b", "bool")
    .ShapeInference([](OpDesignContext * odc){
        auto a = odc->Input(0);
        auto b = odc->Input(1);
        auto y = odc->Output(0);
        bool trans_a = odc->GetAttr<bool>("trans_a");
        bool trans_b = odc->GetAttr<bool>("trans_b");
        std::vector<int> y_shape(2);

        runtime_assert(a.shape().size() == 2,
            "MatMulOp : A is not a matrix.");

        runtime_assert(b.shape().size() == 2,
            "MatMulOp : B is not a matrix.");

        if(trans_a && !trans_b){
            y_shape[0] = a.shape()[1];
            y_shape[1] = b.shape()[1];
        }
        else if(!trans_a && trans_b){
            y_shape[0] = a.shape()[0];
            y_shape[1] = b.shape()[0];
        }
        else if(trans_a && trans_b){
            y_shape[0] = a.shape()[1];
            y_shape[1] = b.shape()[0];
        }
        else{
            y_shape[0] = a.shape()[0];
            y_shape[1] = b.shape()[1];
        }

        y.reshape(y_shape, type::float32());
    })
    .Finish();

REGIST_OP_GRAD(MatMul)
    .Input("A", "float32")
    .Input("B", "float32")
    .Input("Y", "float32")
    .Input("dY", "float32")
    .Output("db", "float32")
    .Output("da", "float32")
    .ShapeInference([](OpDesignContext * odc){
        auto a = odc->Input(0);
        auto b = odc->Input(1);
        auto db = odc->Output(0);
        auto da = odc->Output(1);
        db.reshape(b.shape(), type::float32());
        da.reshape(a.shape(), type::float32());
    })
    .Finish();

class MatMulGradient : public GradientHelper{
public:
    MatMulGradient(const OpDesignContext *odc)
        : GradientHelper(odc){}

    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        Tensor a = y.get_children()[0];
        Tensor b = y.get_children()[1];
        auto ctx = y.get_context();
        bool trans_a = ctx.get_attr<bool>("trans_a");
        bool trans_b = ctx.get_attr<bool>("trans_b");
        if(!trans_a && !trans_b){
            Tensor da = functional::matmul(dy, b, false, true);
            Tensor db = functional::matmul(a, dy, true);
            in_grads.push_back(da);
            in_grads.push_back(db);
        }
        else if(!trans_a && trans_b){
            Tensor da = functional::matmul(dy, b);
            Tensor db = functional::matmul(dy, a, true);
            in_grads.push_back(da);
            in_grads.push_back(db);
        }
        else if(trans_a && !trans_b){
            Tensor da = functional::matmul(b, dy, false, true);
            Tensor db = functional::matmul(a, dy);
            in_grads.push_back(da);
            in_grads.push_back(db);
        }
        else if(trans_a && trans_b){
            Tensor da = functional::matmul(b, dy, true, true);
            Tensor db = functional::matmul(dy, a, true, true);
            in_grads.push_back(da);
            in_grads.push_back(db);
        }
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(MatMul, MatMulGradient)

namespace functional{

Tensor matmul(Tensor a, Tensor b, bool trans_a, bool trans_b){
    Tensor y;
    OpAlgoContext ctx("MatMul");
    std::vector<int> y_shape(2);
    runtime_assert(a.shape().size() == 2,
        "MatMulOp : A is not a matrix.");

    runtime_assert(b.shape().size() == 2,
        "MatMulOp : B is not a matrix.");

    if(trans_a && !trans_b){
        y_shape[0] = a.shape()[1];
        y_shape[1] = b.shape()[1];
    }
    else if(!trans_a && trans_b){
        y_shape[0] = a.shape()[0];
        y_shape[1] = b.shape()[0];
    }
    else if(trans_a && trans_b){
        y_shape[0] = a.shape()[1];
        y_shape[1] = b.shape()[0];
    }
    else{
        y_shape[0] = a.shape()[0];
        y_shape[1] = b.shape()[1];
    }
    y = create_variable(y_shape);
    y.add_child(a);
    y.add_child(b);
    ctx.add_attr({"trans_a", trans_a});
    ctx.add_attr({"trans_b", trans_b});
    Tensor::AssignOpFunctor(y, ctx);

    return y;
}

} // end namespace functional
} // end namespace mlfe
