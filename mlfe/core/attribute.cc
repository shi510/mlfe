#include "attribute.h"
#include <unordered_map>

namespace mlfe{

Attributes::Attributes(){}

bool Attributes::Has(std::string name) const{
    return !(attrs.count(name) <= 0);
}

void Attributes::SetAttr(Attribution attr){
    try{
        Set(attr);
    }
    catch(std::string &e){
        throw e;
    }
}

void Attributes::Set(Attribution attr){
    attrs[attr.Name()] = attr.Attr();
}

std::string Attribution::Name() const{
    return attr.first;
}

Attribution::AttrHolderPtr Attribution::Attr() const{
    return attr.second;
}

struct attribute::pimpl
{
    std::unordered_map<std::string, item> attrs;
};

attribute::attribute() : __p(std::make_shared<pimpl>()){}

void attribute::add(std::string name, item attr_val)
{
    __p->attrs[name] = attr_val;
}

attribute::item attribute::get(std::string key)
{
    if(__p->attrs.find(key) == __p->attrs.end())
    {
        return item();
    }
    return __p->attrs[key];
}

bool attribute::has(std::string key) const
{
    if(__p->attrs.find(key) == __p->attrs.end())
    {
        return false;
    }
    return true;
}

class attribute::item::impl{
public:
    impl();
    ~impl();
    void clear();
    void *_ptr;
    std::function<void(void *)> _destructor;
};

attribute::item::impl::impl():_ptr(nullptr), _destructor(nullptr){}

attribute::item::impl::~impl()
{
    clear();
}

void attribute::item::impl::clear()
{
    if(_ptr != nullptr)
    {
        _destructor(_ptr);
        _ptr = nullptr;
    }
}

attribute::item::item()
{
    init();
}

void attribute::item::init()
{
    _pimpl = std::make_shared<impl>();
}

void attribute::item::set(void *val_ptr, destructor_fn fn)
{
    _pimpl->clear();
    _pimpl->_ptr = val_ptr;
    _pimpl->_destructor = fn;
}

void *attribute::item::get() const
{
    return _pimpl->_ptr;
}

} // end namespace mlfe;
