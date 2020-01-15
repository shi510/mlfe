#include "adam.h"
#include "mlfe/core/op_algo.h"

namespace mlfe{
namespace opt{

class adam : public optimizer{
    using algo_ptr = std::shared_ptr<OpAlgo>;
public:
    adam(double lr, double beta1, double beta2, double eps);

    void apply(Tensor var, Tensor var_grad) override;

private:
    double _lr;
    double _b1;
    double _b2;
    double _eps;
    std::unordered_map<Tensor, algo_ptr> _reg_var;
    std::string _opt_name;
};

adam::adam(double lr, double beta1, double beta2, double eps)
    : _lr(lr), _b1(beta1), _b2(beta2), _eps(eps){
    auto dev = get_enabled_device();
    std::string op_name = "Adam";
    std::string full_op_name = "Name:" + op_name + "/Device:";
    std::string dev_name = dev->get_device_name();
    _opt_name = full_op_name + dev_name;
}

void adam::apply(Tensor var, Tensor var_grad){
    if(_reg_var.find(var) == _reg_var.end()){
        OpAlgoContext oac("Adam");
        oac.add_output(var);
        oac.add_attr({"LearningRate", static_cast<float>(_lr)});
        oac.add_attr({"Beta1", static_cast<float>(_b1)});
        oac.add_attr({"Beta2", static_cast<float>(_b2)});
        oac.add_attr({"Epsilon", static_cast<float>(_eps)});
        _reg_var[var] = OpAlgoRegistry::Get()->GetOpAlgo(_opt_name, &oac);
    }
    _reg_var[var]->Compute();
}

} // end namespace optimizer

namespace functional{

opt::optimizer_ptr create_adam_optimizer(double lr,
                                         double beta1,
                                         double beta2,
                                         double epsilon
                                         ){
    return std::make_shared<opt::adam>(lr, beta1, beta2, epsilon);
}

} // end namespace functional
} // end namespace mlfe
