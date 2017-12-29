#ifndef __ITEM_HOLDER_HPP__
#define __ITEM_HOLDER_HPP__
#include <vector>
#include <memory>
#include <map>

template <class Item>
class ItemHolder{
public:
    bool HasItem(std::string &name) {
        int count = map_item.count(name);
        return count > 0 ? true : false;
    }
    
    template <class ...Args>
    bool AddItem(std::string name, Args... args) {
        if (!HasItem(name)) {
            Item wi_ptr = std::make_shared<Item>(args...);
            map_item[name] = wi_ptr;
        }
        else{
            return false;
        }
        return true;
    }
    
    std::shared_ptr<Item> GetItem(std::string name) {
        auto find = map_item.find(name);
        if (find == map_item.end()) {
            map_item[name] = std::make_shared<Item>();
            return map_item[name];
        }
        return find->second;
    }
    
    int NumItems() {
        return map_item.size();
    }
    
private:
    std::map<std::string, std::shared_ptr<Item>> map_item;
};
#endif /* __ITEM_HOLDER_HPP__ */
