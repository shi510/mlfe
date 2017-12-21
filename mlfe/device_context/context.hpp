#ifndef __CONTEXT_HPP__
#define __CONTEXT_HPP__
#include <memory>
#include <type_traits>
#include <string>

namespace mlfe {

class Context {
public:
    virtual ~Context() {}
    
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
    
    /*
     * @brief Copy from host to device.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToDevice(
                      const unsigned int start,
                      const unsigned int end,
                      const T *host_mem
                      ){
        CopyFrom(start, end, sizeof(T), static_cast<const void *>(host_mem));
    }
    
    /*
     * @brief Copy from device to host.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToHost(
                    const unsigned int start,
                    const unsigned int end,
                    T *host_mem
                    ){
        CopyTo(start, end, sizeof(T), static_cast<void *>(host_mem));
    }
    
    /*
     * @brief Return allocated Device memory byte size.
     */
    virtual int Size() const = 0;
    
    /*
     * @brief Return allocated Device memory address.
     */
    virtual void * GetDevicePtr() const = 0;
    
    struct ComputePrecision {
        using Single = float;
        using Double = double;
    };
    
protected:
    /*
     * @brief Do not allow to instantiate Context class.
     * This class is only for polymorphism design.
     */
    Context() {}
    
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
                          const unsigned int start,
                          const unsigned int end,
                          const unsigned int block_size,
                          const void *from
                          ) = 0;
    
    /*
     * @brief Copy from device memory to host memory.
     * This must be implemented in the inherit class.
     */
    virtual void CopyTo(
                        const unsigned int start,
                        const unsigned int end,
                        const unsigned int block_size,
                        void *to
                        ) = 0;
    
};/* class Context */

} /* namespace mlfe */
#endif /*__CONTEXT_HPP__*/
