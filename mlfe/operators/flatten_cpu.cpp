#include "flatten.hpp"
#include "../device_context/cpu_context.hpp"
#include "../math/blas.hpp"
#include "../math/functions.hpp"
#include "../utils/assert.hpp"

namespace mlfe{

REGIST_OPERATOR_CPU(Flatten_float, FlattenOp<float, CPUContext>)
REGIST_OPERATOR_CPU(Flatten_double, FlattenOp<double, CPUContext>)

REGIST_OPERATOR_CPU(Flatten_float_Gradient, FlattenGradientOp<float, CPUContext>)
REGIST_OPERATOR_CPU(Flatten_double_Gradient, FlattenGradientOp<double, CPUContext>)

struct FlattenGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(Flatten, FlattenGradientIO)

} /* namespace mlfe */
