#include "types.h"

namespace mlfe {

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