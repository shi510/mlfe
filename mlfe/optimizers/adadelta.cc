#include "adadelta.h"
#include "../core/op_algo.h"

namespace mlfe{
namespace opt{

class adadelta : public optimizer{
    using algo_ptr = std::shared_ptr<OpAlgo>;
public:
    adadelta(double lr, double mm, double eps);

    void apply(Tensor var, Tensor var_grad) override;

private:
    double _lr;
    double _mm;
    double _eps;
    std::unordered_map<Tensor, algo_ptr> _reg_var;
    std::string _opt_name;
};

adadelta::adadelta(double lr, double mm, double eps)
    : _lr(lr), _mm(mm), _eps(eps){
    auto dev = get_enabled_device();
    std::string op_name = "AdaDelta";
    std::string full_op_name = "Name:" + op_name + "/Device:";
    std::string dev_name = dev->get_device_name();
    _opt_name = full_op_name + dev_name;
}

void adadelta::apply(Tensor var, Tensor var_grad){
    if(_reg_var.find(var) == _reg_var.end()){
        OpAlgoContext oac("AdaDelta");
        oac.add_output(var);
        oac.add_attr({"LearningRate", static_cast<float>(_lr)});
        oac.add_attr({"Momentum", static_cast<float>(_mm)});
        oac.add_attr({"Epsilon", static_cast<float>(_eps)});
        _reg_var[var] = OpAlgoRegistry::Get()->GetOpAlgo(_opt_name, &oac);
    }
    _reg_var[var]->Compute();
}

} // end namespace optimizer

namespace functional{

opt::optimizer_ptr create_adadelta_optimizer(double lr,
                                             double momentum,
                                             double epsilon
                                             ){
    return std::make_shared<opt::adadelta>(lr, momentum, epsilon);
}

} // end namespace functional
} // end namespace mlfe
