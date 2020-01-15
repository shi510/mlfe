#ifndef __DEVICE_HPP__
#define __DEVICE_HPP__
#include "mlfe/utils/types.h"
#include <memory>
#include <vector>

namespace mlfe{

struct device{
    virtual std::string get_device_name() const = 0;

    virtual std::string get_accelerator_name() const = 0;
};

using device_ptr = std::shared_ptr<device>;

device_ptr get_enabled_device();

// allocate device memory, not host memory.
class memory{
public:
    template <typename T>
    const T *device_data();

    template <typename T>
    T *mutable_device_data();

    template <typename T>
    const T *host_data();

    template <typename T>
    T *mutable_host_data();

    virtual type::uint32::T size() const = 0;

    virtual void allocate(type::uint32::T size) = 0;

    virtual ~memory();

protected:
    virtual const void *_device_data() = 0;

    virtual void *_mutable_device_data() = 0;

    virtual const void *_host_data() = 0;

    virtual void *_mutable_host_data() = 0;

private:
};

template <typename T>
const T *memory::device_data(){
    return static_cast<const T *>(_device_data());
}

template <typename T>
T *memory::mutable_device_data(){
    return static_cast<T *>(_mutable_device_data());
}

template <typename T>
const T *memory::host_data(){
    return static_cast<const T *>(_host_data());
}

template <typename T>
T *memory::mutable_host_data(){
    return static_cast<T *>(_mutable_host_data());
}

using memory_ptr = std::shared_ptr<memory>;

memory_ptr create_memory(type::uint32::T byte_size);

void copy(memory_ptr from, memory_ptr to);

} // end namespace mlfe
#endif // end ifndef __DEVICE_HPP__
