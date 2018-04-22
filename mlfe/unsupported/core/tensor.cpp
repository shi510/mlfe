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

TensorAllocator::TensorAllocator(Accelerator acc, DataType dt)
    : _acc(acc), _dt(dt) {
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

void TensorAllocator::Allocate(unsigned int size, void *ptr) {
    switch (_dt) {
    case DataType::F32:
        _context->Allocate<float>(size, ptr);
        break;
    case DataType::F64:
        _context->Allocate<double>(size, ptr);
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

void TensorAllocator::Type(DataType dt) {
    _dt = dt;
}

void TensorAllocator::Accel(Accelerator acc) {
    _acc = acc;
}

Tensor::Tensor() : trainable(false), bias(false) {}

Tensor::Tensor(std::vector<int> shape) 
    : trainable(false), bias(false)
{
    _shape = std::make_shared<TensorShape>(shape);
    _size = std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
}

Tensor::Tensor(
    std::vector<int> shape,
    void *ptr,
    Accelerator acc, DataType dt)
    : trainable(false), bias(false) {
    _shape = std::make_shared<TensorShape>(shape);
    _ta = std::make_shared<TensorAllocator>(acc, dt);
    _size = std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
    _ta->Allocate(Size(), ptr);
}

Tensor::~Tensor() { }
    
void Tensor::Clear() {
    _shape->Clear();
}

Tensor *Tensor::Reshape(std::vector<int> shape) {
    _shape = std::make_shared<TensorShape>(shape);
    _size = std::accumulate(_shape->Dims().begin(),
        _shape->Dims().end(), 1, std::multiplies<int>());
    return this;
}

void Tensor::Allocate(Accelerator acc, DataType dt) {
    _ta = std::make_shared<TensorAllocator>(acc, dt);
    _ta->Allocate(Size());
}

int Tensor::Size() const {
    return _size;
}

int Tensor::Dims() const{
    return _shape->Dims().size();
}

int Tensor::Dim(int idx) const{
    return _shape->Dims()[idx];
}

std::vector<int> Tensor::Shape() const {
    return _shape->Dims();
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

} // end namespace mlfe
