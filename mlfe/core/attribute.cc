#include "attribute.h"

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

void attribute::add(std::string name, item attr_val)
{
    _attrs[name] = attr_val;
}

attribute::item attribute::get(std::string key)
{
    if(_attrs.find(key) == _attrs.end())
    {
        return item();
    }
    return _attrs[key];
}

bool attribute::has(std::string key) const
{
    if(_attrs.find(key) == _attrs.end())
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
