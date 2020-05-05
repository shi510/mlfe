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
        Tensor x = y.get_children()[0];
        Tensor dx = functional::create_variable(x.shape());
        OpAlgoContext ctx("BatchNormSpatialGradient");
		OpAlgoContext y_ctx = y.get_context();
		ctx.add_attr({ "scales", y_ctx.get_attr<memory_ptr>("scales")});
		ctx.add_attr({ "biases", y_ctx.get_attr<memory_ptr>("biases") });
		ctx.add_attr({ "running_mean", y_ctx.get_attr<memory_ptr>("running_mean") });
		ctx.add_attr({ "running_variance", y_ctx.get_attr<memory_ptr>("running_variance") });
		ctx.add_attr({ "mean", y_ctx.get_attr<memory_ptr>("mean") });
		ctx.add_attr({ "variance", y_ctx.get_attr<memory_ptr>("variance") });
        dx.add_child(dy);
		dx.add_child(x);
        Tensor::AssignOpFunctor(dx, ctx);
        in_grads.push_back(dx);
        return in_grads;
    }
};

REGIST_GRADIENT_HELPER(BatchNormSpatial, BatchNormGradient)

Tensor batch_normalize(Tensor x)
{
	Tensor y = create_variable(x.shape());
	std::string op_name;
	op_name = x.shape().size() == 4 ? "BatchNormSpatial" : "BatchNorm";
	OpAlgoContext ctx(op_name);
	y.add_child(x);
	Tensor::AssignOpFunctor(y, ctx);
	return y;
}

} // end namespace functional
} // end namespace mlfe
