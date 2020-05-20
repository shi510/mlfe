#include "types.h"

namespace mlfe{ namespace type{

TypeInfo::TypeInfo(const std::string type, const unsigned int size)
    : type(type), size(size) {}

DEFINE_TYPE_INFO(uint8, "uint8", sizeof(uint8_t))
DEFINE_TYPE_INFO(uint16, "uint16", sizeof(uint16_t))
DEFINE_TYPE_INFO(uint32, "uint32", sizeof(uint32_t))
DEFINE_TYPE_INFO(int8, "int8", sizeof(int8_t))
DEFINE_TYPE_INFO(int16, "int16", sizeof(uint16_t))
DEFINE_TYPE_INFO(int32, "int32", sizeof(int32_t))
DEFINE_TYPE_INFO(int64, "int64", sizeof(int64_t))
DEFINE_TYPE_INFO(float32, "float32", sizeof(float_t))
DEFINE_TYPE_INFO(float64, "float64", sizeof(double_t))

} // end namespace type
} // end namespace mlfe