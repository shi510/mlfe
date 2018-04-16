#ifndef __ITEM_HOLDER_HPP__
#define __ITEM_HOLDER_HPP__
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace mlfe {
class Item {
public:
    Item() { Clear(); }

    ~Item() { Clear(); }

    Item(const Item &wi) {
        ptr_ = wi.ptr_;
        destructor_ = wi.destructor_;
    }

    Item& operator=(const Item &) = delete;

    template <class T, class ...Args>
    void Set(Args... args) {
        ptr_ = static_cast<void *>(new T(args...));
        destructor_ = Destructor<T>;
    }

    template <class T>
    void Set(T item) {
        ptr_ = static_cast<void *>(&item);
        destructor_ = Destructor<T>;
    }

    template <typename T>
    T * Get() const {
        return static_cast<T*>(ptr_);
    }

    void Clear() {
        if (ptr_ != nullptr && destructor_ != nullptr) {
            destructor_(ptr_);
        }
        ptr_ = nullptr;
        destructor_ = nullptr;
    }

protected:
    template <typename T>
    static void Destructor(void *ptr) {
        delete static_cast<T *>(ptr);
    }

private:
    void *ptr_;
    std::function<void(void *)> destructor_;
};


class ItemHolder {
public:
    bool HasItem(std::string name) {
        int count = map_item.count(name);
        return count > 0 ? true : false;
    }

    template <class Object, class ...Args>
    bool AddItem(std::string name, Args... args) {
        if (!HasItem(name)) {
            std::unique_ptr<Item> item(new Item());
            item->Set<Object>(args...);
            map_item[name] = std::move(item);
        }
        else {
            return false;
        }
        return true;
    }

    template <class NewObject>
    bool AddItem(std::string name, NewObject obj) {
        if (!HasItem(name)) {
            std::unique_ptr<Item> item(new Item());
            item->Set<NewObject>(obj);
            map_item[name] = std::move(item);
        }
        else {
            return false;
        }
        return true;
    }

    template <class Object>
    Object *GetItem(std::string name) {
        auto find = map_item.find(name);
        if (find == map_item.end()) {
            return nullptr;
        }
        return find->second->Get<Object>();
    }
    
    Item *GetItem(std::string name) {
        auto find = map_item.find(name);
        if (find == map_item.end()) {
            return nullptr;
        }
        return find->second.get();
    }
    
    int NumItems() {
        return map_item.size();
    }
    
private:
    std::map<std::string, std::unique_ptr<Item>> map_item;
};
}
#endif /* __ITEM_HOLDER_HPP__ */
