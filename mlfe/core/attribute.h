#pragma once
#include <memory>
#include <map>
#include <string>
#include <functional>

namespace mlfe{
class Attribution;

class Attributes{
public:
    Attributes();

    bool Has(std::string name) const;

    void SetAttr(Attribution attr);

    template <class AttrType>
    AttrType GetAttr(std::string name) const{
        try{
            return Get<AttrType>(name);
        }
        catch(std::string *e){
            throw e;
        }
    }

protected:
    struct AttributeHolderBase{};

    template <class AttrType>
    struct AttributeHolder : AttributeHolderBase{
        AttributeHolder(const AttrType &val) : val(val){}
        AttrType val;
    };

    template <class AttrType>
    AttrType Get(std::string name) const{
        using AT = AttributeHolder<AttrType>;
        if(!Has(name)){
            throw std::string("Attributes.Get : No attribute name -> ") + name;
        }
        return reinterpret_cast<AT *>(attrs.find(name)->second.get())->val;
    }

    void Set(Attribution attr);

private:
    friend class Attribution;
    std::map<std::string, std::shared_ptr<AttributeHolderBase>> attrs;
};

class Attribution{
template <class T>
using AttrHolder = Attributes::AttributeHolder<T>;
using AttrHolderBase = Attributes::AttributeHolderBase;
using AttrHolderPtr = std::shared_ptr<AttrHolderBase>;
using PairAttr = std::pair<std::string, AttrHolderPtr>;
public:
    template <class AttrType>
    Attribution(std::string attr_name, AttrType attr_val){
        auto val_ptr = std::make_shared<AttrHolder<AttrType>>(attr_val);
        attr = std::make_pair(attr_name, val_ptr);
    }

    std::string Name() const;

    AttrHolderPtr Attr() const;

private:
    friend class Attributes;
    PairAttr attr;
};

class attribute final
{
public:
    class item;

    attribute();

    void add(std::string name, item attr_val);

    item get(std::string key);

    bool has(std::string key) const;

private:
    struct pimpl;
    std::shared_ptr<pimpl> __p;
};

class attribute::item final
{
    using destructor_fn = std::function<void(void *)>;

public:
    item();

    template <typename T>
    item(T val);

    template <typename T>
    void assign(T val);

    template <typename T>
    T *data() const;

private:
    void init();

    template <class T>
    static void destruct(void *ptr)
    {
        delete static_cast<T *>(ptr);
    }

    void set(void *val_ptr, destructor_fn fn);

    void *get() const;

private:
    class impl;
    std::shared_ptr<impl> _pimpl;
};

template <typename T>
attribute::item::item(T val)
{
    init();
    assign(val);
}

template <typename T>
void attribute::item::assign(T val)
{
    T *ptr = new T(val);
    set(static_cast<void *>(ptr), destruct<T>);
}

template <typename T>
T *attribute::item::data() const
{
    return static_cast<T *>(get());
}

} // end namespace mlfe
