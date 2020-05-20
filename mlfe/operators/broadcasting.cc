#include "broadcasting.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

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

Tensor broadcast(Tensor x,
                 std::vector<type::int32::T> shape
                 ){
    auto x_shape = x.shape();
    auto bc_shape = math::check_broadcasting(&x_shape, &shape);
    if (bc_shape.empty())
    {
        std::stringstream ss;
        ss << "Can not broadcast from ";
        for (auto& v : x.shape()) ss << v << " ";
        ss << "to ";
        for (auto& v : shape) ss << v << " ";
        ss << std::endl;
        throw std::runtime_error(ss.str());
    }
    Tensor y = create_variable(bc_shape);
    OpAlgoContext ctx("Broadcasting");
    ctx.add_input(x);
    ctx.add_output(y);
    ctx.add_attr({ "broadcasting_shape", shape });
    y.set_context(ctx);
    return y;
}

} // end namespace functional
} // end namespace mlfe
