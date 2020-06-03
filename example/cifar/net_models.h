#pragma once
#include <vector>
#include <mlfe/module/model.h>

namespace models
{

mlfe::module::model conv_dropout_net(std::vector<int> input_shape);

mlfe::module::model conv_bn_net(std::vector<int> input_shape);

} // end namespace models