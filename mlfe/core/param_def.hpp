#ifndef __PARAM_DEF_HPP__
#define __PARAM_DEF_HPP__
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace mlfe{
    
struct ParamContainerBase {};

template <typename ValType>
struct ParamValueContainer : public ParamContainerBase {
    ParamValueContainer(const ValType& value) : v(value) {}
    ValType v;
};

class ParamDef {
public:
    bool HasParam(std::string name){
        return params.at(name).use_count() > 0 ? true : false;
    }
    
    /*
     * returns true when parameter's key exists.
     * if not, returns false.
     */
    template <class T>
    T GetParam(std::string name){
        T t;
        try {
            t = static_cast<ParamValueContainer<T> *>(params.at(name).get())->v;
            return t;
        }
        catch (std::out_of_range e) {
            throw std::string("No Param -> ") + name;
        }
    }
    
    /*
     * receive any type of data exept for the type of const char *.
     */
    template<class T,
    typename = typename std::enable_if<!std::is_same<T, const char *>::value>::type
    >
    ParamDef Add(std::string key, T value) {
        AddParam(key, value);
        return *this;
    }
    
    /*
     * receive only the type of string.
     */
    ParamDef Add(std::string key, std::string value) {
        AddParam(key, value);
        return *this;
    }
    
    void Clear(){
        params.clear();
    }
    
protected:
    
    /*
     * add parameter's name and value.
     */
    template <class T>
    bool AddParam(std::string name, T &value) {
        params.insert(std::pair<std::string, std::shared_ptr<ParamContainerBase>>(name,
                                                                                  std::make_shared<ParamValueContainer<T>>(value)));
        return true;
    }
    
private:
    std::map<std::string, std::shared_ptr<ParamContainerBase>> params;
};

} /* namespace mlfe */
#endif /* __PARAM_DEF_HPP__ */
