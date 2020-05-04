#include "optimizer.h"

namespace mlfe{
namespace opt{

optimizer::optimizer(double lr){
    _lr = functional::create_variable({1});
    _lr.mutable_data<float>()[0] = lr;
}

void optimizer::update_learning_rate(double lr){
    _lr.mutable_data<float>()[0] = lr;
}

double optimizer::get_learning_rate(){
    return static_cast<double>(_lr.data<float>()[0]);
}

} // end namespace optimizer
} // end namespace mlfe
