#pragma once

// include layers
#include "mlfe/module/layers/batch_norm.h"
#include "mlfe/module/layers/input.h"
#include "mlfe/module/layers/dense.h"
#include "mlfe/module/layers/dropout.h"
#include "mlfe/module/layers/relu.h"
#include "mlfe/module/layers/sigmoid.h"
#include "mlfe/module/layers/conv2d.h"
#include "mlfe/module/layers/maxpool2d.h"
#include "mlfe/module/layers/flatten.h"

// include callbacks
#include "mlfe/module/callbacks/tensorboard.h"
#include "mlfe/module/callbacks/reduce_lr.h"
// include model
#include "mlfe/module/model.h"