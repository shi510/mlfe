#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include "../../utils/type_holder.hpp"
#include "../../device_context/context.hpp"
#include "../utils/types.hpp"

namespace mlfe{

class TensorShape {
public:
    TensorShape();

    TensorShape(std::vector<int> dims);

    const std::vector<int> &Dims() const;

    void Clear();

private:
    std::vector<int> _dims;
};

class TensorAllocator {
public:
    TensorAllocator();

    TensorAllocator(DataType dt, Accelerator acc);

    TensorAllocator(const TensorAllocator &ta) = default;

    void Allocate(unsigned int size);

    void CopyTo(TensorAllocator &ta) const;

    void CopyFrom(const TensorAllocator &ta);

    template <typename T>
    T *GetPtr() {
        return reinterpret_cast<T *>(_context->GetDevicePtr());
    }

    DataType Type() const;

    Accelerator Accel() const;

private:
    DataType _dt;
    Accelerator _acc;
    std::shared_ptr<Context> _context;
};

class Tensor final{
public:
    explicit Tensor(
        Accelerator acc = Accelerator::Default,
        DataType dt = DataType::F32
    );

    explicit Tensor(
        std::vector<int> shape, 
        Accelerator acc = Accelerator::Default,
        DataType dt = DataType::F32
    );

    ~Tensor();

    Tensor(const Tensor &t) = default;

    Tensor *Reshape(std::vector<int> shape);

    void Allocate();

    void Clear();

    int Size() const;

    int Dims() const;

    int Dim(int idx) const;

    void CopyFrom(const Tensor &t);

    void CopyTo(Tensor &t) const;

    DataType Type() const;

    Accelerator Accel() const;

    void SetTrainable(bool val);

    void SetBias(bool val);

    bool IsTrainable() const;

    bool IsBias() const;

    template <typename T>
    T *GetPtr() {
        return _ta->GetPtr<T>();
    }
    
private:
    std::shared_ptr<TensorShape> _shape;
    std::shared_ptr<TensorAllocator> _ta;
    bool trainable;
    bool bias;
};

std::ostream &operator<<(std::ostream &os, Tensor &t);

} // end namespace mlfe
#endif // end ifndef __TENSOR_HPP__
