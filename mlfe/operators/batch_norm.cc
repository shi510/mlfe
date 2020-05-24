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
		Tensor scales = ctx_y.get_input(1);
		Tensor biases = ctx_y.get_input(2);
        Tensor dx = functional::create_variable(x.shape());
		Tensor dscales = functional::create_variable(scales.shape());
		Tensor dbiases = functional::create_variable(biases.shape());
        OpAlgoContext ctx_dx("BatchNormSpatialGradient");
		ctx_dx.set_attrs(ctx_y.get_attrs());
		ctx_dx.add_input(dy);
		ctx_dx.add_input(x);
		ctx_dx.add_input(scales);
		ctx_dx.add_input(biases);
		ctx_dx.add_output(dx);
		ctx_dx.add_output(dscales);
		ctx_dx.add_output(dbiases);
		dx.set_context(ctx_dx);
		dscales.set_context(ctx_dx);
		dbiases.set_context(ctx_dx);
		x.set_backprop_node(dx.get_node());
		scales.set_backprop_node(dscales.get_node());
		biases.set_backprop_node(dbiases.get_node());
		x.set_gradient(dx);
		scales.set_gradient(dscales);
		biases.set_gradient(dbiases);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(BatchNormSpatial, BatchNormGradient)

Tensor batch_normalize(Tensor x)
{
	std::vector<int> norm_dim = { x.shape()[1] };
	Tensor y = create_variable(x.shape());
	Tensor scales = create_variable(norm_dim);
	Tensor biases = create_variable(norm_dim);
	OpAlgoContext ctx("BatchNormSpatial");
	ctx.add_input(x);
	ctx.add_input(scales);
	ctx.add_input(biases);
	ctx.add_output(y);
	y.set_context(ctx);
	return y;
}

Tensor batch_normalize(Tensor x, Tensor scales, Tensor biases)
{
	Tensor y = create_variable(x.shape());
	Tensor mean = create_variable(scales.shape());
	Tensor var = create_variable(scales.shape());
	OpAlgoContext ctx("BatchNormSpatial");
	std::fill(mean.begin<float>(), mean.end<float>(), 0.f);
	std::fill(var.begin<float>(), var.end<float>(), 1.f);
	ctx.add_input(x);
	ctx.add_input(scales);
	ctx.add_input(biases);
	ctx.add_output(y);
	ctx.add_attr({ "running_mean", mean });
	ctx.add_attr({ "running_var", var });
	y.set_context(ctx);
	return y;
}

Tensor batch_normalize(Tensor x, Tensor scales, Tensor biases,
	Tensor mean, Tensor var)
{
	Tensor y = create_variable(x.shape());
	OpAlgoContext ctx("BatchNormSpatial");
	ctx.add_input(x);
	ctx.add_input(scales);
	ctx.add_input(biases);
	ctx.add_output(y);
	ctx.add_attr({ "running_mean", mean });
	ctx.add_attr({ "running_var", var });
	y.set_context(ctx);
	return y;
}

} // end namespace functional
} // end namespace mlfe
