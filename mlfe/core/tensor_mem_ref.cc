#include "tensor_mem_ref.h"

namespace mlfe{

TensorMemRef::TensorMemRef(const Tensor &t, DeviceMemory dev_mem)
    : ref(t), dev_mem(dev_mem){
    dev_mem.Allocate(ref.Size() * ref.Type().size);
}

size_t TensorMemRef::Size() const{
    return ref.Size();
}

std::vector<int> TensorMemRef::Shape() const{
    return ref.Shape();
}

DeviceMemory TensorMemRef::GetDeviceMemory() const{
    return dev_mem;
}

} // end namespace mlfe;
