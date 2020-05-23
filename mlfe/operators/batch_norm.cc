#include "batch_norm.h"
#include "mlfe/core/op_registry.h"
#include "mlfe/core/op_algo.h"
#include "mlfe/core/gradient_helper.h"

namespace mlfe{
namespace functional{

class BatchNormGradient : public GradientHelper{
public:
    VecTensor compute_gradient(Tensor y, Tensor dy) override{
        using IntVec = std::vector<type::int32::T>;
        VecTensor in_grads;
		auto ctx_y = y.get_context();
        Tensor x = ctx_y.get_input(0);
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx_dx("BatchNormSpatialGradient");
		ctx_dx.add_attr({ "scales", ctx_y.get_attr<memory_ptr>("scales")});
		ctx_dx.add_attr({ "biases", ctx_y.get_attr<memory_ptr>("biases") });
		ctx_dx.add_attr({ "running_mean", ctx_y.get_attr<memory_ptr>("running_mean") });
		ctx_dx.add_attr({ "running_variance", ctx_y.get_attr<memory_ptr>("running_variance") });
		ctx_dx.add_attr({ "mean", ctx_y.get_attr<memory_ptr>("mean") });
		ctx_dx.add_attr({ "variance", ctx_y.get_attr<memory_ptr>("variance") });
		ctx_dx.add_input(dy);
		ctx_dx.add_input(x);
		ctx_dx.add_output(dx);
		dx.set_context(ctx_dx);
		x.set_backprop_node(dx.get_node());
		x.set_gradient(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(BatchNormSpatial, BatchNormGradient)

Tensor batch_normalize(Tensor x)
{
	Tensor y = create_variable(x.shape());
	OpAlgoContext ctx("BatchNormSpatial");
	ctx.add_input(x);
	ctx.add_output(y);
	y.set_context(ctx);
	return y;
}

} // end namespace functional
} // end namespace mlfe
