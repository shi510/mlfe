#include <string>
#include <new>
#include <functional>
#include "cpu_context.hpp"

namespace mlfe {

CPUContext::CPUContext() : Context(Accelerator::Default){
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


void CPUContext::Allocator(
                           const unsigned int size,
                           const unsigned int block_size
                           ){
    Clear();
    size_ = size * block_size;
    try {
        ptr_ = static_cast<void *>(new char[size_]);
    }
    catch (std::bad_alloc &e) {
        throw std::string("CPUContext::Allocator() : ") + std::string(e.what());
    }
}

void CPUContext::Allocator(
    const unsigned int size,
    const unsigned int block_size,
    void *ptr
) {
    Clear();
    size_ = size * block_size;
    ptr_ = ptr;
}

void CPUContext::CopyTo(
                        const unsigned int offset,
                        const unsigned int size,
                        const unsigned int block_size,
                        void *to
                        ) const{
    if((offset + size) * block_size > size_){
        throw std::string("Copy size is bigger than allocated device memory.");
    }
    const unsigned char *from_ = static_cast<const unsigned char *>(ptr_);
    unsigned char *to_ = static_cast<unsigned char *>(to);
    for (int i = 0; i < size * block_size; ++i) {
        to_[i] = from_[i + offset * block_size];
    }
}

void CPUContext::CopyFrom(
                          const unsigned int offset,
                          const unsigned int size,
                          const unsigned int block_size,
                          const void *from
                          ){
    if((offset + size) * block_size > size_){
        throw std::string("Copy size is bigger than allocated device memory.");
    }
    const unsigned char *from_ = static_cast<const unsigned char *>(from);
    unsigned char *to_ = static_cast<unsigned char *>(ptr_);
    for (int i = 0; i < size * block_size; ++i) {
        to_[i + offset * block_size] = from_[i];
    }
}

REGIST_CONTEXT(Context_Default, CPUContext)

struct Cpu2CpuCopyFunctor : ContextSwitchCopier {
    void copy(
        const std::shared_ptr<Context> src,
        std::shared_ptr<Context> dst) override {
        dst->CopyToDevice(0, src->Size(), (unsigned char *)(src->GetDevicePtr()));
    }
};

REGIST_CONTEXT_SWITCH_COPY(Context_Copy_Default_Default, Cpu2CpuCopyFunctor)
    
} /* namespace mlfe */
