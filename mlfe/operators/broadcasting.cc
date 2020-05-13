#include "broadcasting.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"
#include <algorithm>

namespace mlfe{
namespace functional{

class BroadcastingGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        VecTensor in_grads;
        auto ctx_y = y.get_context();
        auto x = ctx_y.get_input(0);
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx_x_grad("BroadcastingGradient");
        ctx_x_grad.add_input(dy);
        ctx_x_grad.add_input(y);
        ctx_x_grad.add_output(dx);
        dx.set_context(ctx_x_grad);
        x.set_backprop_node(dx.get_node());
        x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(Broadcasting, BroadcastingGradient)

std::vector<int> check_broadcasting(std::vector<int> a,
                                    std::vector<int> b
                                   ){
    std::vector<int> shape;
    int max = std::max(a.size(), b.size());
    while(max != a.size()){
        a.insert(a.begin(), 1);
    }
    while(max != b.size()){
        b.insert(b.begin(), 1);
    }
    shape.resize(max);
    for(int n = max - 1; n >= 1; --n){
        int a_at = a[n];
        int b_at = b[n];
        if(a_at != 1 && b_at != 1 && a_at != b_at){
            throw std::string("Can not broadcasting.");
        }
        else{
            shape[n] = std::max(a_at, b_at);
        }
    }
    shape[0] = std::max(a[0], b[0]);
    return shape;
}

Tensor broadcast(Tensor x,
                 std::vector<type::int32::T> shape
                 ){
    check_broadcasting(x.shape(), shape);
    Tensor y = create_variable(shape);
    OpAlgoContext ctx("Broadcasting");
    ctx.add_input(x);
    ctx.add_output(y);
    ctx.add_attr({ "broadcasting_shape", shape });
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
