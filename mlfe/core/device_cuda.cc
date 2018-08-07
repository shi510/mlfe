#include "device.h"
#include <cuda_runtime.h>

namespace mlfe{
using CUDADevice = Device::Select<Device::CUDA>;

template <>
struct CUDADevice::DeviceData{
    DeviceData(){}

    DeviceData(type::uint8::T *ptr, type::uint32::T size)
        : data(ptr), size(size){}

    type::uint8::T *data;
    type::uint32::T size;
};

template<>
CUDADevice::Select(){
    dd = std::make_shared<DeviceData>();
}

template<>
CUDADevice::Select(type::uint8::T *ptr, type::uint32::T size){
    throw std::string("Device::Select<Device::CUDA>() - "
        "Feeding address by direct is not supported.");
}

template<>
type::uint8::T *CUDADevice::Data() const{
    return dd->data;
}

template <>
type::uint32::T CUDADevice::Size() const{
    return dd->size;
}

template<>
void CUDADevice::Allocate(type::uint32::T size){
    dd->size = size;
    if(cudaMalloc((void**)&dd->data, size) != cudaSuccess){
        throw std::string("Device::Select<Device::CUDA>::Allocate() - "
            "device memory allocation failed.");
    }
}

template <>
std::string CUDADevice::Name() const{
    return Device::CUDA::string;
}

template <> void
Device::CopyInternal<Device::CUDA, Device::CPU>(const Device from, Device to){
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyDeviceToHost) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CUDA, Device::CPU>() "
            ": Copy from cuda to cpu is failed.";
        throw err;
    }
}

template <> void
Device::CopyInternal<Device::CPU, Device::CUDA>(const Device from, Device to){
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyHostToDevice) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CPU, Device::CUDA>() "
            ": Copy from cpu to cuda is failed.";
        throw err;
    }
}

template <> void
Device::CopyInternal<Device::CUDA, Device::CUDA>(const Device from, Device to){
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyDeviceToDevice) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CUDA, Device::CUDA>() "
            " : Copy from cuda to cuda is failed.";
        throw err;
    }
}

const char *Device::CUDA::string = "Device.CUDA";

} // end namespace mlfe;
