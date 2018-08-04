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
    TensorMemRef(const Tensor &t, Device device);

    size_t Size() const;

    std::vector<int> Shape() const;

    template <class T>
    T *Data() const;

    Device GetDevice() const;

private:
    const Tensor ref;
    Device d;
};

template <class T>
T *TensorMemRef::Data() const{
    return d.Data<T>();
}

} // end namespace mlfe
#endif // end ifndef __TENSOR_MEM_REF_HPP__
