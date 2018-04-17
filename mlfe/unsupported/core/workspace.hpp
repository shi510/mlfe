#ifndef __WORKSPACE_HPP__
#define __WORKSPACE_HPP__
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "tensor.hpp"
#include "../utils/types.hpp"
#include "../../core/item_holder.hpp"

namespace mlfe {

class Workspace final{
public:
    Workspace();

    Workspace(const Workspace &) = delete;

    Workspace &operator=(const Workspace &) = delete;

    template <typename Object, typename ...Args>
    Object *Create(std::string name, Args ...args)
    {
        auto item = std::make_shared<Item>();
        item->Set<Object>(args...);
        items[name] = item;
        return item->Get<Object>();
    }

    template <typename Object>
    Object *Get(std::string name) {
        return items[name]->Get<Object>();
    }

    template <typename Object>
    Object *GetIfNotExistCreate(std::string name) {
        if (items[name].use_count() <= 0) {
            Create<Object>(name);
        }
        return items[name]->Get<Object>();
    }

private:
    std::map<std::string, std::shared_ptr<Item>> items;
};
} // end namespace mlfe
#endif // end ifndef __WORKSPACE_HPP__
