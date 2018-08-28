#include "device.h"
#include <vector>

namespace mlfe{
using CPUDevice = Device::Select<Device::CPU>;

class CpuMemory : public DeviceMemory::Allocator{
public:
    CpuMemory();

    ~CpuMemory() override;

    type::uint8::T *Data() const override;

    void Allocate(type::uint32::T size) override;

    void Allocate(type::uint8::T *ptr, type::uint32::T size) override;

    type::uint32::T Size() const override;

private:
    struct DeviceData;
    std::shared_ptr<DeviceData> dd;
};

struct CpuMemory::DeviceData{
    std::vector<type::uint8::T> data;
    std::pair<type::uint8::T *, type::uint32::T> external_ptr;
};

CpuMemory::CpuMemory(){
    dd = std::make_shared<DeviceData>();
}

CpuMemory::~CpuMemory(){
    dd->data.clear();
}

type::uint8::T *CpuMemory::Data() const{
    return dd->external_ptr.first == nullptr ?
        dd->data.data() : dd->external_ptr.first;
}

void CpuMemory::Allocate(type::uint32::T size){
    dd->data.resize(size);
    if(Size() != size){
        throw std::string("CPUDevice::Allocate - memory allocating failed.");
    }
    dd->external_ptr.first = nullptr;
    dd->external_ptr.second = 0;
}

void CpuMemory::Allocate(type::uint8::T *ptr, type::uint32::T size){
    dd->external_ptr.first = ptr;
    dd->external_ptr.second = size;
    dd->data.clear();
}

type::uint32::T CpuMemory::Size() const{
    return dd->external_ptr.first == nullptr ?
        dd->data.size() : dd->external_ptr.second;
}

template<>
DeviceMemory CPUDevice::CreateDeviceMemory() const{
    return DeviceMemory(std::make_shared<CpuMemory>());
}

template <>
std::string CPUDevice::Name() const{
    return Device::CPU::string;
}

template <> void
Device::CopyInternal<Device::CPU, Device::CPU>(const DeviceMemory from,
                                               DeviceMemory to
                                              )
{
    auto from_ptr = from.Data();
    auto to_ptr = to.Data();
    auto size = from.Size();
    for(int n = 0; n < size; ++n){
        to_ptr[n] = from_ptr[n];
    }
}

} // end namespace mlfe;
