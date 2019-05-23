#ifndef __MLFE_OP_REGISTRY_H__
#define __MLFE_OP_REGISTRY_H__
#include "mlfe/core/op.h"
#include <unordered_map>
#include <memory>
#include <string>

namespace mlfe{
namespace detail{

struct op_info{
    using creator = std::unique_ptr<op>(*)();
    creator fn;
    std::string desc;
};

class op_registry{
    using reg_type = std::unordered_map<std::string, op_info>;
public:
    op_registry() = delete;

    static reg_type & get();

    static bool regist(std::string name, op_info info);

    static std::unique_ptr<op> create(const std::string name);
};

template <typename T>
class registerer{
public:
    class access{
    public:
        static std::string get_name(){
            return T().name;
        }

        static std::string get_description(){
            return T().description;
        }
    };

    static std::unique_ptr<op> create();

protected:
    static bool is_registered;
};

template <typename T>
std::unique_ptr<op> registerer<T>::create(){
    return std::make_unique<T>();
}

template <typename T>
bool registerer<T>::is_registered =
    op_registry::regist(registerer<T>::access::get_name(), 
                        {T::create, registerer<T>::access::get_description()});

template <typename Derived>
class op_impl : public op, public registerer<Derived>{
protected:
    friend class registerer<Derived>::access;
    std::string name;
    std::string description;
};

} // end namespace detail
} // end namespace mlfe

#endif // end #ifndef __MLFE_OP_REGISTRY_H__