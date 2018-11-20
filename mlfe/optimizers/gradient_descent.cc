#include "gradient_descent.h"
#include "../core/tensor.h"
#include "../core/op_algo.h"
#include "../math/basic_functions.h"
#include <unordered_map>

namespace mlfe{
namespace opt{

class gradient_descent : public optimizer{
    using algo_ptr = std::shared_ptr<OpAlgo>;
public:
    gradient_descent(double lr, double momentum);

    void apply(Tensor var, Tensor var_grad) override;

private:
    double _lr;
    double _mm;
    std::unordered_map<Tensor, algo_ptr> _reg_var;
    std::string _opt_name;
};

gradient_descent::gradient_descent(double lr, double momentum)
    : _lr(lr), _mm(momentum){
    auto dev = get_enabled_device();
    std::string op_name = "GradientDescent";
    std::string full_op_name = "Name:" + op_name + "/Device:";
    std::string dev_name = dev->get_device_name();
    _opt_name = full_op_name + dev_name;
}

void gradient_descent::apply(Tensor var, Tensor var_grad){
    if(_reg_var.find(var) == _reg_var.end()){
        OpAlgoContext oac("GradientDescent");
        oac.add_output(var);
        oac.add_attr({"LearningRate", static_cast<float>(_lr)});
        oac.add_attr({"MomentumRate", static_cast<float>(_mm)});
        oac.add_attr({"WeightDecay", static_cast<float>(0)});
        _reg_var[var] = OpAlgoRegistry::Get()->GetOpAlgo(_opt_name, &oac);
    }
    _reg_var[var]->Compute();
}

} // end namespace optimizer

namespace functional{

opt::optimizer_ptr create_gradient_descent_optimizer(double lr, 
                                                     double momentum
                                                     ){
    return std::make_shared<opt::gradient_descent>(lr, momentum);
}

} // end namespace functional
} // end namespace mlfe