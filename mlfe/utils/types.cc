#include "types.h"

namespace mlfe{ namespace type{

TypeInfo::TypeInfo(const std::string type, const unsigned int size)
    : type(type), size(size) {}

DEFINE_TYPE_INFO(uint8, "uint8", 1U)
DEFINE_TYPE_INFO(uint16, "uint16", 2U)
DEFINE_TYPE_INFO(uint32, "uint32", 4U)
DEFINE_TYPE_INFO(int8, "int8", 1U)
DEFINE_TYPE_INFO(int16, "int16", 2U)
DEFINE_TYPE_INFO(int32, "int32", 4U)
DEFINE_TYPE_INFO(float32, "float32", 4U)
DEFINE_TYPE_INFO(float64, "float64", 8U)

} // end namespace type

std::string to_string(DataType dt) {
    std::string str;
    switch (dt) {
    case DataType::F32:
        str = "F32";
        break;
    case DataType::F64:
        str = "F64";
        break;
    }
    return str;
}

std::string to_string(Accelerator acc) {
    std::string str;
    switch (acc) {
    case Accelerator::Default:
        str = "Default";
        break;
    case Accelerator::CUDA:
        str = "Cuda";
        break;
    case Accelerator::CUDNN:
        str = "Cudnn";
        break;
    case Accelerator::OpenCL:
        str = "Opencl";
        break;
    }
    return str;
}

} // end namespace mlfe