#ifndef __TENSOR_MEM_REF_HPP__
#define __TENSOR_MEM_REF_HPP__
#include "tensor.h"
#include "device.h"
#include <functional>
#include <memory>

namespace mlfe{

// TODO : Remove TensorMemRef class.
class TensorMemRef{
public:
    TensorMemRef(const Tensor &t, DeviceMemory mem);

    size_t Size() const;

    std::vector<int> Shape() const;

    template <class T>
    T *Data() const;

    DeviceMemory GetDeviceMemory() const;

private:
    const Tensor ref;
    DeviceMemory dev_mem;
};

template <class T>
T *TensorMemRef::Data() const{
    return reinterpret_cast<T *>(dev_mem.Data());
}

} // end namespace mlfe
#endif // end ifndef __TENSOR_MEM_REF_HPP__
