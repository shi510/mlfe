#ifndef __ATTRIBUTE_HPP__
#define __ATTRIBUTE_HPP__
#include <memory>
#include <map>
#include <string>

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

} // end namespace mlfe
#endif // end ifndef __ATTRIBUTE_HPP__
