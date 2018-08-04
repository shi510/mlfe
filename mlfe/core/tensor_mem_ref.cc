#include "tensor_mem_ref.h"

namespace mlfe{

TensorMemRef::TensorMemRef(const Tensor &t, Device device)
    : ref(t), d(device){
    d.Allocate(ref.Size() * ref.Type().size);
}

size_t TensorMemRef::Size() const{
    return ref.Size();
}

std::vector<int> TensorMemRef::Shape() const{
    return ref.Shape();
}

Device TensorMemRef::GetDevice() const{
    return d;
}

} // end namespace mlfe;
