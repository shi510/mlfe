#include "device.h"

namespace mlfe{

DeviceMemory Device::CreateDeviceMemory() const{
    return sd->CreateDeviceMemory();
}

std::string Device::Name() const{
    return sd->Name();
}

DeviceMemory::DeviceMemory(){}

DeviceMemory::DeviceMemory(std::shared_ptr<Allocator> alloc){
    this->alloc = alloc;
}

type::uint8::T *DeviceMemory::Data() const{
    return alloc->Data();
}

void DeviceMemory::Allocate(type::uint32::T size){
    alloc->Allocate(size);
}

void DeviceMemory::Allocate(type::uint8::T *ptr, type::uint32::T size){
    alloc->Allocate(ptr, size);
}

type::uint32::T DeviceMemory::Size() const{
    return alloc->Size();
}

DeviceMemory::Allocator::~Allocator(){}

const char *Device::CPU::string = "Device.CPU";
const char *Device::CUDA::string = "Device.CUDA";
const char *Device::CUDA::string_cudnn = "Device.CUDA(CUDNN)";
bool Device::CUDA::enable_cudnn = false;

} // end namespace mlfe;