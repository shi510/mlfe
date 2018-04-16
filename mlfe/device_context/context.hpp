#ifndef __CONTEXT_HPP__
#define __CONTEXT_HPP__
#include <memory>
#include <type_traits>
#include <string>
#include "../core/registry.hpp"
#include "../unsupported/utils/types.hpp"

namespace mlfe {

class Context {
public:
    Context(Accelerator acc);

    virtual ~Context();
    
    /*
     * @brief Allocate device memory.
     * The inherit classes of Context should hold allocated memory.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void Allocate(const int size) {
        try {
            Allocator(size, sizeof(T));
        }
        catch (std::string &e) {
            throw e;
        }
    }

    static std::shared_ptr<Context> Create(Accelerator acc);

    void Allocate(const int size, const int block_size);

    static void Copy(
        const std::shared_ptr<Context> src, 
        std::shared_ptr<Context> dst
    );

    virtual void Clear() = 0;
    
    /*
     * @brief Copy from host to device.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToDevice(
                      const unsigned int offset,
                      const unsigned int size,
                      const T *host_mem
                      ){
        CopyFrom(offset, size, sizeof(T), static_cast<const void *>(host_mem));
    }
    
    /*
     * @brief Copy from device to host.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToHost(
                    const unsigned int offset,
                    const unsigned int size,
                    T *host_mem
                    ) const{
        CopyTo(offset, size, sizeof(T), static_cast<void *>(host_mem));
    }
    
    /*
     * @brief Return allocated Device memory byte size.
     */
    virtual int Size() const = 0;
    
    /*
     * @brief Return allocated Device memory address.
     */
    virtual void * GetDevicePtr() const = 0;

protected:
    
    /*
     * @brief Device specific memory allocator.
     * This must be implemented in the inherit class.
     */
    virtual void Allocator(
                           const unsigned int size,
                           const unsigned int block_size
                           ) = 0;
    
    /*
     * @brief Copy from host memory to device memory.
     * This must be implemented in the inherit class.
     */
    virtual void CopyFrom(
                          const unsigned int offset,
                          const unsigned int size,
                          const unsigned int block_size,
                          const void *from
                          ) = 0;
    
    /*
     * @brief Copy from device memory to host memory.
     * This must be implemented in the inherit class.
     */
    virtual void CopyTo(
                        const unsigned int offset,
                        const unsigned int size,
                        const unsigned int block_size,
                        void *to
                        ) const = 0;

    // member variables
    std::string acc_str;
};/* class Context */

struct ContextSwitchCopier {
    virtual void copy(
        const std::shared_ptr<Context> src, 
        std::shared_ptr<Context> dst) = 0;
};


// TODO : remove ContextCopyRegistry.
DECLARE_REGISTRY(
    ContextSwitchCopyRegistry,
    std::string,
    std::shared_ptr<ContextSwitchCopier>
)

#define REGIST_CONTEXT_SWITCH_COPY(Key, ...)                  \
namespace {   \
static RegistererContextSwitchCopyRegistry (ContextSwitchCopyRegistry_##Key)(      \
  #Key,                                                \
  ContextSwitchCopyRegistry(),                                       \
  RegistererContextSwitchCopyRegistry::DefaultCreator<__VA_ARGS__>   \
);                                                     \
} // end namespace


// TODO : remove ContextRegistry.
DECLARE_REGISTRY(
    ContextRegistry,
    std::string,
    std::shared_ptr<Context>
)

#define REGIST_CONTEXT(Key, ...)                  \
namespace {   \
static RegistererContextRegistry (ContextRegistry_##Key)(      \
  #Key,                                                \
  ContextRegistry(),                                       \
  RegistererContextRegistry::DefaultCreator<__VA_ARGS__>   \
);                                                     \
} // end namespace

} /* namespace mlfe */
#endif /*__CONTEXT_HPP__*/
