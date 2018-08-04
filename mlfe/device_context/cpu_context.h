#ifndef __CPU_CONTEXT_HPP__
#define __CPU_CONTEXT_HPP__
#include <functional>
#include "context.h"

namespace mlfe {
    
class CPUContext final : public Context {
public:
    CPUContext();
    
    ~CPUContext() override;
    
    void * GetDevicePtr() const override;
    
    void Clear() override;
    
    int Size() const override;
    
protected:
    void Allocator(
                   const unsigned int size,
                   const unsigned int block_size
                   ) override;

    void Allocator(
                   const unsigned int size,
                   const unsigned int block_size,
                   void *ptr
    ) override;
    
    void CopyTo(
                const unsigned int offset,
                const unsigned int size,
                const unsigned int block_size,
                void *to
                ) const override;
    
    void CopyFrom(
                  const unsigned int offset,
                  const unsigned int size,
                  const unsigned int block_size,
                  const void *from
                  ) override;
    
    template <typename T>
    static void Destructor(void *_ptr) {
        delete[] static_cast<T *>(_ptr);
    }
    
private:
    int size_;
    void *ptr_;
    std::function<void(void *)> destructor_;
};/* class CPUContext */
    
} /* namespace mlfe */
#endif /*__CPU_CONTEXT_HPP__*/
