#include "device.h"
#include <vector>

namespace mlfe{
using CPUDevice = Device::Select<Device::CPU>;

template <>
struct CPUDevice::DeviceData{
    DeviceData(){
        external_ptr.first = nullptr;
        external_ptr.second = 0;
    }

    // TODO: Remove external_ptr.
    DeviceData(type::uint8::T *ptr, type::uint32::T size){
        external_ptr.first = ptr;
        external_ptr.second = size;
    }

    std::vector<type::uint8::T> data;
    std::pair<type::uint8::T *, type::uint32::T> external_ptr;
};

template<>
CPUDevice::Select(){
    dd = std::make_shared<DeviceData>();
}

template<>
CPUDevice::Select(type::uint8::T *ptr, type::uint32::T size){
    dd = std::make_shared<DeviceData>(ptr, size);
}

template <>
type::uint8::T *CPUDevice::Data() const{
    return dd->external_ptr.first == nullptr ?
        dd->data.data() : dd->external_ptr.first;
}

template <>
type::uint32::T CPUDevice::Size() const{
    return dd->external_ptr.first == nullptr ?
        dd->data.size() : dd->external_ptr.second;
}

template <>
void CPUDevice::Allocate(type::uint32::T size){
    dd->data.resize(size);
    if(Size() != size){
        throw std::string("CPUDevice::Allocate - memory allocating failed.");
    }
}

template <>
std::string CPUDevice::Name() const{
    return Device::CPU::string;
}

template <> void
Device::CopyInternal<Device::CPU, Device::CPU>(const Device from, Device to){
    auto from_ptr = from.Data();
    auto to_ptr = to.Data();
    auto size = from.Size();
    for(int n = 0; n < size; ++n){
        to_ptr[n] = from_ptr[n];
    }
}

const char *Device::CPU::string = "Device.CPU";

} // end namespace mlfe;
