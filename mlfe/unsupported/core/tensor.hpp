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

class TensorShape final{
public:
    TensorShape();

    TensorShape(std::vector<int> dims);

    const std::vector<int> &Dims() const;

    void Clear();

private:
    std::vector<int> _dims;
};

class TensorAllocator final{
public:
    TensorAllocator();

    TensorAllocator(Accelerator acc, DataType dt);

    TensorAllocator(const TensorAllocator &ta) = default;

    void Allocate(unsigned int size);

    void Allocate(unsigned int size, void *ptr);

    void CopyTo(TensorAllocator &ta) const;

    void CopyFrom(const TensorAllocator &ta);

    template <typename T>
    T *GetPtr() {
        return reinterpret_cast<T *>(_context->GetDevicePtr());
    }

    DataType Type() const;

    Accelerator Accel() const;

    void Type(DataType dt);

    void Accel(Accelerator acc);

private:
    DataType _dt;
    Accelerator _acc;
    std::shared_ptr<Context> _context;
};

class Tensor final{
public:
    Tensor();

    explicit Tensor(std::vector<int> shape);

    explicit Tensor(
        std::vector<int> shape,
        void *ptr,
        Accelerator acc = Accelerator::Default,
        DataType dt = DataType::F32
    );

    ~Tensor();

    Tensor(const Tensor &t);

    Tensor *Reshape(std::vector<int> shape);

    void Allocate(
        Accelerator acc = Accelerator::Default,
        DataType dt = DataType::F32
    );

    void Clear();

    int Size() const;

    int Dims() const;

    int Dim(int idx) const;
    
    std::vector<int> Shape() const;

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
    int _size;
    bool trainable;
    bool bias;
};

} // end namespace mlfe
#endif // end ifndef __TENSOR_HPP__
