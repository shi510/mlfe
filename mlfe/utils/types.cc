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
} // end namespace mlfe