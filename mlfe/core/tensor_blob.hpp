#ifndef __TENSOR_BLOB_HPP__
#define __TENSOR_BLOB_HPP__
#include <string>
#include <vector>
#include <memory>
#include "../device_context/context.hpp"
#include "../utils/type_holder.hpp"

namespace mlfe{

template <class DeviceContext,
class = typename std::enable_if<std::is_base_of<Context, DeviceContext>::value, DeviceContext>::type
>
class TensorBlob{
public:
    TensorBlob() : size(0), context(std::make_shared<DeviceContext>()){}
    
    ~TensorBlob() { Clear(); }
    
    TensorBlob(const TensorBlob &) = delete;
    
    TensorBlob& operator=(const TensorBlob &tb){
        dims = tb.dims;
        size = tb.size;
        context = tb.context;
        type = tb.type;
        return *this;
    }
    
    void Reshape(const std::vector<int> new_dims){
        int new_size = 1;
        
        dims.clear();
        for(int n = 0; n < new_dims.size(); ++n){
            new_size *= new_dims[n];
            dims.push_back(new_dims[n]);
        }
        if(new_size != size){
            throw std::string("reshape size does not match.");
        }
        size = new_size;
    }
    
    void Reshape(const TensorBlob<DeviceContext> &tb){
        Reshape(tb.dims);
    }
    
    /*
     * @brief reshape tensor's shape.
     */
    template <typename T,
    class = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void Resize(const std::vector<int> new_dims){
        int new_size = 1;
        
        dims.clear();
        for(int n = 0; n < new_dims.size(); ++n){
            new_size *= new_dims[n];
            dims.push_back(new_dims[n]);
        }
        if(new_size > size){
            size = new_size;
            context->Allocate<T>(size);
        }
        else{
            size = new_size;
        }
        type.Set<T>();
    }
    
    template <typename T,
    class = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void Resize(const TensorBlob<DeviceContext> &tb) {
        std::vector<int> new_size;
        
        for (int i = 0; i < tb.dims.size(); ++i) {
            new_size.push_back(tb.dims[i]);
        }
        Resize<T>(new_size);
    }
    
    /*
     * @brief compare tensor's size.
     * if same then returns true, or not returns false.
     */
    bool CompareSizeWith(const TensorBlob<DeviceContext> &tb){
        if(this->Dims() != tb.Dims()){
            return false;
        }
        for(int i = 0; i < this->Dims(); ++i){
            if(this->Dim(i) != tb.Dim(i)){
                return false;
            }
        }
        return true;
    }
    
    /*
     * @brief check empty.
     */
    bool IsEmpty(){
        return (size == 0 ? true : false);
    }
    
    /*
     * @brief clear tensor data.
     */
    void Clear() {
        size = 0;
        dims.clear();
    }
    
    /*
     * @brief returns total size.
     */
    int Size() const {
        return size;
    }
    
    /*
     * @brief returns number of dimensions.
     */
    int Dims() const {
        return dims.size();
    }
    
    /*
     * @brief returns dimension size.
     */
    int Dim(int idx) const {
        return dims[idx];
    }
    
    /*
     * @brief returns const tensor data address.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    const T * GetPtrConst() const{
        return static_cast<const T *>(context->GetDevicePtr());
    }
    
    /*
     * @brief returns tensor data address.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    T * GetPtrMutable() const{
        return static_cast<T *>(context->GetDevicePtr());
    }
    
    /*
     * @brief copy data from device to host.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToHost(
                    const unsigned int start,
                    const unsigned int end,
                    T *host_mem
                    ){
        context->CopyToHost<T>(start, end, host_mem);
    }
    
    /*
     * @brief copy data from host to device.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void CopyToDevice(
                      const unsigned int start,
                      const unsigned int end,
                      const T *host_mem
                      ){
        context->CopyToDevice<T>(start, end, host_mem);
    }
    
    template <class T>
    bool MatchType(){
        return type.Id() == TypeHolder::Id<T>();
    }
    
    /*
     * @brief set all tensor's elements by const value.
     */
    template <typename T,
    typename = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void SetByConst(const T val){
        T * data_ptr = static_cast<T *>(context->GetDevicePtr());
        for(int i = 0; i < size; ++i){
            data_ptr[i] = val;
        }
    }
    
private:
    std::vector<int> dims;
    int size;
    std::shared_ptr<Context> context;
    TypeHolder type;
};

} /* namespace mlfe */
#endif /* __TENSOR_BLOB_HPP__ */
