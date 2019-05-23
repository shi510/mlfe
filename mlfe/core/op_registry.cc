#include "op_registry.h"

namespace mlfe{
namespace detail{

op_registry::reg_type & op_registry::get(){
    static reg_type reg_infos;
    return reg_infos;
}

bool op_registry::regist(std::string name,
                         op_info info){
	auto it = get().find(name);
    if(it == get().end()){
        get()[name] = info;
        return true;
    }
    return false;
}

std::unique_ptr<op> op_registry::create(const std::string name){
	auto it = get().find(name);
    if(it != get().end()){
        return it->second.fn();
    }
    return nullptr;
}

} // end namespace detail
} // end namespace mlfe
