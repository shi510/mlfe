#include "mlfe/operators/broadcast.h"
#include "mlfe/math/transform.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace mlfe{
namespace operators{

Tensor broadcast(Tensor x, std::vector<int32_t> shape)
{
    if (x.dims() > 4 || shape.size() > 4) {
        throw std::runtime_error("broadcasting only supports upto 4 dimensions.");
    }
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
    auto y = functional::create_variable(bc_shape);
    auto gm_x = [=](Tensor &dy){
        broadcast_bwd_kernel::fn(dy, x.grad());
    };
    call<broadcast_fwd_kernel>(
        marker::I(x),
        marker::O(y)(gm_x));
    return y;
}

} // namespace operators
} // namespace mlfe
