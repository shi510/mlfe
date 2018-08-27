#include "device.h"
#include <cuda_runtime.h>

namespace mlfe{
using CUDADevice = Device::Select<Device::CUDA>;

class CudaMemory : public DeviceMemory::Allocator{
public:
    CudaMemory();

    ~CudaMemory() override;

    type::uint8::T *Data() const override;

    void Allocate(type::uint32::T size) override;

    void Allocate(type::uint8::T *ptr, type::uint32::T size) override;

    type::uint32::T Size() const override;

private:
    struct DeviceData;
    std::shared_ptr<DeviceData> dd;
};

struct CudaMemory::DeviceData{
    DeviceData() : data(nullptr), size(0){}
    type::uint8::T *data;
    type::uint32::T size;
};

CudaMemory::CudaMemory(){
    dd = std::make_shared<DeviceData>();
}

CudaMemory::~CudaMemory(){
    dd->size = 0;
    if(dd->data != nullptr){
        if(cudaFree(dd->data) != cudaSuccess){
            throw std::string("Device::Select<Device::CUDA>::~Select() - "
                "device memory free failed.");
        }
    }
}

type::uint8::T *CudaMemory::Data() const{
    return dd->data;
}

void CudaMemory::Allocate(type::uint32::T size){
    dd->size = size;
    if(cudaMalloc((void**)&dd->data, size) != cudaSuccess){
        throw std::string("Device::Select<Device::CUDA>::Allocate() - "
            "device memory allocation failed.");
    }
}

void CudaMemory::Allocate(type::uint8::T *ptr, type::uint32::T size){
    throw std::string("Device::Select<Device::CUDA>() - "
        "Feeding address by direct is not supported.");
}

type::uint32::T CudaMemory::Size() const{
    return dd->size;
}

template<>
DeviceMemory CUDADevice::CreateDeviceMemory() const{
    return DeviceMemory(std::make_shared<CudaMemory>());
}

template <>
std::string CUDADevice::Name() const{
    return Device::CUDA::string;
}

template <> void
Device::CopyInternal<Device::CUDA, Device::CPU>(const DeviceMemory from,
                                                DeviceMemory to
                                               )
{
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyDeviceToHost) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CUDA, Device::CPU>() "
            ": Copy from cuda to cpu is failed.";
        throw err;
    }
}

template <> void
Device::CopyInternal<Device::CPU, Device::CUDA>(const DeviceMemory from,
                                                DeviceMemory to
                                               )
{
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyHostToDevice) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CPU, Device::CUDA>() "
            ": Copy from cpu to cuda is failed.";
        throw err;
    }
}

template <> void
Device::CopyInternal<Device::CUDA, Device::CUDA>(const DeviceMemory from,
                                                 DeviceMemory to
                                                )
{
    if(cudaMemcpy(to.Data(), from.Data(), from.Size(),
        cudaMemcpyDeviceToDevice) != cudaSuccess){
        const std::string err =
            "Device::CopyInternal<Device::CUDA, Device::CUDA>() "
            " : Copy from cuda to cuda is failed.";
        throw err;
    }
}

const char *Device::CUDA::string = "Device.CUDA";
const char *Device::CUDA::string_cudnn = "Device.CUDA(CUDNN)";
bool Device::CUDA::enable_cudnn = false;

} // end namespace mlfe;
