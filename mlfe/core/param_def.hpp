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
    template<class T>
    explicit ParamDef(std::string key, T value) {
        if (key != "" && !key.empty()) {
            operator()(key, value);
        }
    }
    
    ParamDef() {}
    
    ~ParamDef() {}
    
    /*
     * returns true when parameter's key exists.
     * if not, returns false.
     */
    template <class T>
    bool GetParamByName(std::string key, T &value) {
        try {
            value = static_cast<ParamValueContainer<T> *>(params.at(key).get())->v;
        }
        catch (std::out_of_range e) {
            return false;
        }
        return true;
    }
    
    /*
     * receive any type of data exept for the type of const char *.
     */
    template<class T,
    typename = typename std::enable_if<!std::is_same<T, const char *>::value>::type
    >
    ParamDef operator()(std::string key, T value) {
        AddParam(key, value);
        return *this;
    }
    
    /*
     * receive only the type of string.
     */
    ParamDef operator()(std::string key, std::string value) {
        AddParam(key, value);
        return *this;
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
