#include "matmul.h"
#include "../core/op_algo.h"
#include "../core/gradient_helper.h"
#include "../utils/assert.h"

namespace mlfe{

class MatMulGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        VecTensor in_grads;
        auto ctx = y.get_context();
        auto a = ctx.get_input(0);
        auto b = ctx.get_input(1);
        bool trans_a = ctx.get_attr<bool>("trans_a");
        bool trans_b = ctx.get_attr<bool>("trans_b");
        Tensor da, db;
        da.set_name(a.name() + "_d");
        db.set_name(b.name() + "_d");
        if (!trans_a && !trans_b) {
            da = functional::matmul(dy, b, false, true);
            db = functional::matmul(a, dy, true);
        }
        else if (!trans_a && trans_b) {
            da = functional::matmul(dy, b);
            db = functional::matmul(dy, a, true);
        }
        else if (trans_a && !trans_b) {
            da = functional::matmul(b, dy, false, true);
            db = functional::matmul(a, dy);
        }
        else if (trans_a && trans_b) {
            da = functional::matmul(b, dy, true, true);
            db = functional::matmul(dy, a, true, true);
        }
        a.set_backprop_node(da.get_node());
        b.set_backprop_node(db.get_node());
        a.set_gradient(da);
        b.set_gradient(db);
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

    if (trans_a && !trans_b) {
        y_shape[0] = a.shape()[1];
        y_shape[1] = b.shape()[1];
    }
    else if (!trans_a && trans_b) {
        y_shape[0] = a.shape()[0];
        y_shape[1] = b.shape()[0];
    }
    else if (trans_a && trans_b) {
        y_shape[0] = a.shape()[1];
        y_shape[1] = b.shape()[0];
    }
    else {
        y_shape[0] = a.shape()[0];
        y_shape[1] = b.shape()[1];
    }
    y = functional::create_variable(y_shape);
    ctx.add_input(a);
    ctx.add_input(b);
    ctx.add_output(y);
    ctx.add_attr({ "trans_a", trans_a });
    ctx.add_attr({ "trans_b", trans_b });
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
