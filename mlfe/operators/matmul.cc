#include "matmul.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "../utils/assert.h"

namespace mlfe{

class MatMulGradient : public GradientHelper{
public:
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
