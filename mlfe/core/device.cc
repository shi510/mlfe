#include "device.h"
#include <functional>
#include <cstring>
#if defined(OPTION_USE_CUDNN) || defined(OPTION_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace mlfe{

struct enabled_device final : device{
    enabled_device();

    std::string get_device_name() const override;

    std::string get_accelerator_name() const override;

    std::string _dev_name;
    std::string _accel_name;
};

enabled_device::enabled_device(){
#if defined(OPTION_USE_CUDNN)
    _dev_name = "CUDA";
    _accel_name = "CUDNN";
#elif defined(OPTION_USE_CUDA)
    _dev_name = "CUDA";
    _accel_name = "NONE";
#elif defined(OPTION_USE_MKLDNN)
    _dev_name = "CPU";
    _accel_name = "MKLDNN";
#else
    _dev_name = "CPU";
    _accel_name = "NONE";
#endif
}

std::string enabled_device::get_device_name() const{
    return _dev_name;
}

std::string enabled_device::get_accelerator_name() const{
    return _accel_name;
}

device_ptr get_enabled_device(){
    static device_ptr dev = std::make_shared<enabled_device>();
    return dev;
}

memory::~memory(){}

class device_memory final : public memory{
public:
    void allocate(type::uint32::T size) override;

    type::uint32::T size() const override;

    ~device_memory() override;

protected:
    const void *_device_data() override;

    void *_mutable_device_data() override;

    const void *_host_data() override;

    void *_mutable_host_data() override;

private:
    void *_h_data;
    void *_d_data;
    type::uint32::T _byte_size;
    bool is_mutated_host;
    bool is_mutated_device;
};

// for nvidia cuda device memory synchronization.
#if defined(OPTION_USE_CUDNN) || defined(OPTION_USE_CUDA)

void sync_d2h(void *to, 
              const void *from, 
              type::uint32::T size
             ){
    if(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost) != cudaSuccess){
        const std::string err = "device_memory::sync_d2h - failed.";
        throw err;
    }
}

void sync_h2d(void *to,
              const void *from,
              type::uint32::T size
             ){
    if(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice) != cudaSuccess){
        const std::string err = "device_memory::sync_h2d() - failed.";
        throw err;
    }
}

void sync_d2d(void *to,
              const void *from,
              type::uint32::T size
             ){
    if(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice) != cudaSuccess){
        const std::string err = "device_memory::sync_d2d() - failed.";
        throw err;
    }
}

void device_memory::allocate(type::uint32::T byte_size){
    _byte_size = byte_size;
    if(cudaMalloc((void**)&_d_data, _byte_size) != cudaSuccess){
        throw std::string("device_memory::allocate() - "
            "failed to allocate cuda memory.");
    }
    _h_data = static_cast<void *>(new type::uint8::T[_byte_size]);
    if(_h_data == nullptr){
        throw std::string("device_memory::allocate() - "
            "failed to allocate host memory.");
    }
}

device_memory::~device_memory(){
    _byte_size = 0;
    if(_d_data != nullptr){
        if(cudaFree(_d_data) != cudaSuccess){
            throw std::string("device_memory::~device_memory() - "
                "failed to free memory.");
        }
        _d_data = nullptr;
    }
    if(_h_data != nullptr){
        delete[] static_cast<type::uint8::T *>(_h_data);
        _h_data = nullptr;
    }
}

// for cpu memory synchronization.
#else

void sync_d2h(void *to, 
              const void *from, 
              type::uint32::T size
             ){}

void sync_h2d(void *to,
              const void *from,
              type::uint32::T size
             ){}

void sync_d2d(void *to,
              const void *from,
              type::uint32::T size
             ){
    memcpy(to, from, size);
}

void device_memory::allocate(type::uint32::T byte_size){
    _byte_size = byte_size;
    _h_data = static_cast<void *>(new type::uint8::T[_byte_size]);
    if(_h_data == nullptr){
        throw std::string("device_memory::allocate() - "
            "failed to allocate host memory.");
    }
    _d_data = _h_data;
}

device_memory::~device_memory(){
    _byte_size = 0;
    if(_h_data != nullptr){
        delete[] static_cast<type::uint8::T *>(_h_data);
        _h_data = nullptr;
        _d_data = nullptr;
    }
}

#endif

type::uint32::T device_memory::size() const{
    return _byte_size;
}

const void *device_memory::_device_data(){
    if(is_mutated_host){
        sync_h2d(_d_data, _h_data, _byte_size);
        is_mutated_host = false;
    }

    return const_cast<const void *>(_d_data);
}

void *device_memory::_mutable_device_data(){
    if(is_mutated_host){
        sync_h2d(_d_data, _h_data, _byte_size);
        is_mutated_host = false;
    }
    is_mutated_device = true;

    return _d_data;
}

const void *device_memory::_host_data(){
    if(is_mutated_device){
        sync_d2h(_h_data, _d_data, _byte_size);
        is_mutated_device = false;
    }
    return static_cast<const void *>(_h_data);
}

void *device_memory::_mutable_host_data(){
    if(is_mutated_device){
        sync_d2h(_h_data, _d_data, _byte_size);
        is_mutated_device = false;
    }
    is_mutated_host = true;

    return _h_data;
}

memory_ptr create_memory(type::uint32::T byte_size){
    memory_ptr mem = std::make_shared<device_memory>();
    mem->allocate(byte_size);
    return mem;
}

void copy(memory_ptr from, memory_ptr to){
    if(from->size() != to->size()){
        throw std::string("copy() - size not matches");
    }
    sync_d2d(to->mutable_device_data<void>(), 
             from->device_data<void>(), 
             from->size()
            );
}

} // end namespace mlfe;