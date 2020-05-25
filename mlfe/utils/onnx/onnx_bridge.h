#pragma once
#include "mlfe/utils/onnx/proto/onnx.proto3.pb.h"
#include "mlfe/module/module.h"
#include <string>

namespace mlfe{
namespace onnx{
using namespace ::onnx;

bool export_onnx_model(module::model m, std::string onnx_file_name);

} // end namesapce onnx
} // end namespace mlfe