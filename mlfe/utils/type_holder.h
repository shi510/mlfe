#ifndef __TYPE_HOLDER_HPP__
#define __TYPE_HOLDER_HPP__
#include <string>
#include <typeinfo>

namespace mlfe{

class TypeHolder{
public:
    template <class T>
    void Set(){
        type_name = std::string(typeid(T).name());
    }
    
    template <class T>
    static std::string Id(){
        return std::string(typeid(T).name());
    }
    
    std::string Id(){
        return type_name;
    }
    
private:
    std::string type_name;
};

} /* namespace mlfe */

#endif /* __TYPE_HOLDER_HPP__ */
