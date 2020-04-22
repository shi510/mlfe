#pragma once
#include <vector>
#include <mlfe/module/module.h>

namespace models
{

mlfe::module::model simple_net(std::vector<int> input_shape);

mlfe::module::model conv_net(std::vector<int> input_shape);

mlfe::module::model auto_encoder(std::vector<int> input_shape);

} // end namespace models