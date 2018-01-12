#ifndef __REGISTRY_HPP__
#define __REGISTRY_HPP__
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <iostream>

namespace mlfe{

template <class Key, class Val, class... Args>
class Registry {
public:
    typedef std::function<Val(Args...)> Creator;
    
    Registry() : registry_() {}
    
    void Register(const Key& key, Creator creator) {
        if (registry_.count(key) != 0) {
            printf("Key already registered.\n");
            std::exit(1);
        }
        registry_[key] = creator;
    }
    
    void Register(const Key& key, Creator creator, const std::string& help_msg) {
        Register(key, creator);
    }
    
    inline bool Has(const Key& key) { return (registry_.count(key) != 0); }
    
    Val Create(const Key& key, Args... args) {
        if (registry_.count(key) == 0) {
            return nullptr;
        }
        return registry_[key](args...);
    }
    
    std::vector<Key> Keys() {
        std::vector<Key> keys;
        for (const auto& it : registry_) {
            keys.push_back(it.first);
        }
        return keys;
    }
    
private:
    std::map<Key, Creator> registry_;
};

template <class Key, class Val, class... Args>
class Registerer {
public:
    Registerer(
               const Key key,
               Registry<Key, Val, Args...>* registry,
               typename Registry<Key, Val, Args...>::Creator creator
               ) {
        registry->Register(key, creator);
    }
    
    template <class DerivedType>
    static Val DefaultCreator(Args... args) {
        return Val(new DerivedType(args...));
    }
};

#define DECLARE_REGISTRY(                                    \
Name, Key, Val, ...)                     \
Registry<Key, Val, ##__VA_ARGS__> *Name();  \
typedef Registerer<Key, Val, ##__VA_ARGS__>        \
Registerer##Name;
    
#define DEFINE_REGISTRY(                                         \
Name, Key, Val, ...)                         \
Registry<Key, Val, ##__VA_ARGS__> *Name() {     \
static Registry<Key, Val, ##__VA_ARGS__> *registry = \
new Registry<Key, Val, ##__VA_ARGS__>();         \
return registry;                                                         \
}

} /* namespace mlfe */
#endif /* __REGISTRY_HPP__ */
