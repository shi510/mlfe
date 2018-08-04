#ifndef __TENSOR_BLOB_HPP__
#define __TENSOR_BLOB_HPP__
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include "../device_context/context.h"
#include "../utils/type_holder.h"

namespace mlfe{

template <class DeviceContext,
class = typename std::enable_if<std::is_base_of<Context, DeviceContext>::value, DeviceContext>::type
>
class TensorBlob {
public:
    TensorBlob() : size(0), block_size(0), context(std::make_shared<DeviceContext>()){}
    
    ~TensorBlob() { Clear(); }
    
    TensorBlob(const TensorBlob &) = default;

    template <class OtherContext>
    TensorBlob(const TensorBlob<OtherContext> &tb) :
        context(std::make_shared<DeviceContext>()) {
        *this = tb;
    }
    
    TensorBlob& operator=(const TensorBlob &tb){
        dims = tb.dims;
        size = tb.size;
        block_size = tb.block_size;
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
        block_size = sizeof(T);
    }

    void Resize(const std::vector<int> new_dims, const int block_size) {
        int new_size = 1;

        dims.clear();
        for (int n = 0; n < new_dims.size(); ++n) {
            new_size *= new_dims[n];
            dims.push_back(new_dims[n]);
        }
        if (new_size > size || context->Size() == 0) {
            size = new_size;
            context->Allocate(size, block_size);
        }
        else {
            size = new_size;
        }
        this->block_size = block_size;
    }
    
    template <typename T,
        typename OtherContext,
        class = typename std::enable_if<std::is_fundamental<T>::value, T>::type
    >
    void Resize(const TensorBlob<OtherContext> &tb) {
        std::vector<int> new_size;
        
        for (int i = 0; i < tb.Dims(); ++i) {
            new_size.push_back(tb.Dim(i));
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

    int BlockSize() const {
        return block_size;
    }
    
    /*
     * @brief returns number of dimensions.
     */
    int Dims() const {
        return dims.size();
    }
    
    std::vector<int> Dim_() const {
        return dims;
    }
    
    /*
     * @brief returns dimension size.
     */
    int Dim(int idx) const {
        return dims[idx];
    }

    TypeHolder Type() const {
        return type;
    }

    std::shared_ptr<Context> GetContext() const{
        return context;
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

public:
    
private:
    std::vector<int> dims;
    int size;
    int block_size;
    std::shared_ptr<Context> context;
    TypeHolder type;
};

} /* namespace mlfe */
#endif /* __TENSOR_BLOB_HPP__ */
