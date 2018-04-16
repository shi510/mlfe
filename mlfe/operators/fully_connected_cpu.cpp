#include "fully_connected.hpp"
#include "../device_context/cpu_context.hpp"

namespace mlfe{

REGIST_OPERATOR_CPU(FC_float, FullyConnectedOp<float, CPUContext>)
REGIST_OPERATOR_CPU(FC_double, FullyConnectedOp<double, CPUContext>)

REGIST_OPERATOR_CPU(FC_float_Gradient, FullyConnectedGradientOp<float, CPUContext>)
REGIST_OPERATOR_CPU(FC_double_Gradient, FullyConnectedGradientOp<double, CPUContext>)

struct FCGradientIO : public GradientIO{
    OperatorIO GetGradientIO(OperatorIO opio) override{
        OperatorIO opio_grad;
        opio_grad.type = opio.type + "_" + opio.data_type + "_Gradient";
        opio_grad.data_type = opio.data_type;
        opio_grad.inputs.push_back(opio.inputs[0]);
        opio_grad.inputs.push_back(opio.inputs[1]);
        opio_grad.inputs.push_back(opio.outputs[0] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[1] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[2] + "_grad");
        opio_grad.outputs.push_back(opio.inputs[0] + "_grad");
        opio_grad.param = opio.param;
        
        return opio_grad;
    }
};

REGIST_OPERATOR_GRADIENT_IO(FC, FCGradientIO)

} /* namespace mlfe */
