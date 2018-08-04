#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <string>

#define __NAME_CONCAT(x, y) x##y
#define NAME_CONCAT(x, y) __NAME_CONCAT(x, y)

namespace mlfe {
enum class DataType {
    F32, F64
};

enum class Accelerator {
    Default, OpenCL, CUDA, CUDNN
};

std::string to_string(DataType dt);

std::string to_string(Accelerator acc);

template <typename T>
T to_value(std::string val_str) {
    double val = atof(val_str.c_str());
    return static_cast<T>(val);
}

} // end namespace mlfe
#endif // end ifndef __UTILS_HPP__