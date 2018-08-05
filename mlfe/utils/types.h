#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <string>

#define __NAME_CONCAT(x, y) x##y
#define NAME_CONCAT(x, y) __NAME_CONCAT(x, y)

#define DECLARE_TYPE_INFO(TypeName, TType) \
    struct TypeName : TypeInfo{            \
        TypeName();                        \
        using T = TType;                   \
        static const char *string;         \
        static const unsigned int size;    \
    };

#define DEFINE_TYPE_INFO(TypeName, String, Size)    \
    TypeName::TypeName() : TypeInfo(string, size){} \
    const char *type::TypeName::string = String;    \
    const unsigned int type::TypeName::size = Size;

namespace mlfe{ namespace type{

    struct TypeInfo{
        TypeInfo(const std::string type, const unsigned int size);
        std::string type;
        unsigned int size;
    };

    template <class To, class From>
    To Cast(From from){
        return reinterpret_cast<To>(from);
    }

    DECLARE_TYPE_INFO(uint8, unsigned char)
    DECLARE_TYPE_INFO(uint16, unsigned short)
    DECLARE_TYPE_INFO(uint32, unsigned int)
    DECLARE_TYPE_INFO(int8, char)
    DECLARE_TYPE_INFO(int16, short)
    DECLARE_TYPE_INFO(int32, int)
    DECLARE_TYPE_INFO(float32, float)
    DECLARE_TYPE_INFO(float64, double)

} // end namespace type

template <typename T>
T to_value(std::string val_str) {
    double val = atof(val_str.c_str());
    return static_cast<T>(val);
}

} // end namespace mlfe
#endif // end ifndef __UTILS_HPP__