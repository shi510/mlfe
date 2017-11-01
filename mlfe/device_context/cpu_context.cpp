#include <string>
#include <new>
#include <functional>
#include "cpu_context.hpp"

namespace mlfe {

CPUContext::CPUContext() {
    Clear();
}
CPUContext::~CPUContext() {
    Clear();
}

void * CPUContext::GetDevicePtr() const {
    return ptr_;
}

void CPUContext::Clear() {
    if (ptr_ != nullptr && destructor_ != nullptr) {
        destructor_(ptr_);
    }
    size_ = 0;
    ptr_ = nullptr;
    destructor_ = nullptr;
}

int CPUContext::Size() const {
    return size_;
}


void CPUContext::Allocator(const int size, const int block_size) {
    Clear();
    size_ = size * block_size;
    try {
        ptr_ = static_cast<void *>(new char[size_]);
    }
    catch (std::bad_alloc &e) {
        throw std::string("CPUContext::Allocator() : ") + std::string(e.what());
    }
}

void CPUContext::CopyTo(const int size, const int block_size, void *to) {
    if (size * block_size > size_) {
        throw std::string("Copy size is bigger than allocated device memory.");
    }
    const unsigned char *from_ = static_cast<const unsigned char *>(ptr_);
    unsigned char *to_ = static_cast<unsigned char *>(to);
    for (int i = 0; i < size_; ++i) {
        to_[i] = from_[i];
    }
}

void CPUContext::CopyFrom(const int size, const int block_size, const void *from) {
    if (size * block_size > size_) {
        throw std::string("Copy size is bigger than allocated device memory.");
    }
    const unsigned char *from_ = static_cast<const unsigned char *>(from);
    unsigned char *to_ = static_cast<unsigned char *>(ptr_);
    for (int i = 0; i < size_; ++i) {
        to_[i] = from_[i];
    }
}
    
} /* namespace mlfe */
