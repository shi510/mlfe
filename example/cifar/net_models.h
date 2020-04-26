#pragma once
#include <vector>
#include <mlfe/module/model.h>

namespace models
{

mlfe::module::model conv_net(std::vector<int> input_shape);

} // end namespace models