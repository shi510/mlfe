#include "device.h"

namespace mlfe{

void Device::Allocate(type::uint32::T size){
    sd->Allocate(size);
}

type::uint8::T *Device::Data() const{
    return sd->Data();
}

type::uint32::T Device::Size() const{
    return sd->Size();
}

std::string Device::Name() const{
    return sd->Name();
}

} // end namespace mlfe;