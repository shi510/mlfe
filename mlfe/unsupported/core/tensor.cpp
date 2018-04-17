#include "tensor.hpp"
#include "../../utils/assert.hpp"
#include <iomanip>

namespace mlfe {

TensorShape::TensorShape() {}

TensorShape::TensorShape(std::vector<int> dims) {
    _dims = dims;
}

const std::vector<int> &TensorShape::Dims() const {
    return _dims;
}

void TensorShape::Clear() {
    _dims.clear();
}

TensorAllocator::TensorAllocator()
    : _acc(Accelerator::Default), _dt(DataType::F32){
    _context = Context::Create(_acc);
}

TensorAllocator::TensorAllocator(DataType dt, Accelerator acc)
    : _dt(dt), _acc(acc){
    _context = Context::Create(_acc);
}

void TensorAllocator::Allocate(unsigned int size) {
    switch (_dt) {
    case DataType::F32:
        _context->Allocate<float>(size);
        break;
    case DataType::F64:
        _context->Allocate<double>(size);
        break;
    }
}

void TensorAllocator::CopyTo(TensorAllocator &ta) const {
    Context::Copy(_context, ta._context);
}

void TensorAllocator::CopyFrom(const TensorAllocator &ta) {
    Context::Copy(ta._context, _context);
}

DataType TensorAllocator::Type() const {
    return _dt;
}

Accelerator TensorAllocator::Accel() const {
    return _acc;
}

Tensor::Tensor(Accelerator acc, DataType dt) 
    : trainable(false), bias(false) {
    _shape = std::make_shared<TensorShape>();
    _ta = std::make_shared<TensorAllocator>(dt, acc);
}

Tensor::Tensor(
    std::vector<int> shape,
    Accelerator acc, DataType dt) 
    : trainable(false), bias(false){
    _shape = std::make_shared<TensorShape>(shape);
    _ta = std::make_shared<TensorAllocator>(dt, acc);
    _ta->Allocate(Size());
}

Tensor::~Tensor() { }
    
void Tensor::Clear() {
    _shape->Clear();
}

Tensor *Tensor::Reshape(std::vector<int> shape) {
    *_shape = TensorShape(shape);
    return this;
}

void Tensor::Allocate() {
    _ta->Allocate(Size());
}

int Tensor::Size() const {
    return std::accumulate(_shape->Dims().begin(), 
        _shape->Dims().end(), 1, std::multiplies<int>());
}

int Tensor::Dims() const{
    return _shape->Dims().size();
}

int Tensor::Dim(int idx) const{
    return _shape->Dims()[idx];
}

void Tensor::CopyFrom(const Tensor &t) {
    _ta->CopyFrom(*t._ta);
}

void Tensor::CopyTo(Tensor &t) const {
    _ta->CopyTo(*t._ta);
}

DataType Tensor::Type() const{
    return _ta->Type();
}

Accelerator Tensor::Accel() const{
    return _ta->Accel();
}

void Tensor::SetTrainable(bool val) {
    trainable = val;
}

void Tensor::SetBias(bool val) {
    bias = val;
}

bool Tensor::IsTrainable() const {
    return trainable;
}

bool Tensor::IsBias() const {
    return bias;
}

std::ostream &operator<<(std::ostream &os, Tensor &t) {
    const float *ptr = t.GetPtr<float>();
    const int dim0 = t.Size() / t.Dim(0);
    for (int n = 0; n < t.Size(); ++n) {
        os << std::fixed << std::setw(9) << std::setprecision(6) << std::setfill(' ') << ptr[n];
        if ((n + 1) == dim0) {
            os << std::endl;
        }
    }
    os << std::endl;
    return os;
}

} // end namespace mlfe
