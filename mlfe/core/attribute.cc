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
    if(Has(attr.Name())){
        throw std::string("Attributes.Set : Attribute already registered. -> ") + attr.Name();
    }
    attrs[attr.Name()] = attr.Attr();
}

std::string Attribution::Name() const{
    return attr.first;
}

Attribution::AttrHolderPtr Attribution::Attr() const{
    return attr.second;
}

} // end namespace mlfe;
