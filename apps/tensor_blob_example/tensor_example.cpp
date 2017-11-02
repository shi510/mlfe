#include <iostream>
#include <mlfe/core/tensor_blob.hpp>
#include <mlfe/device_context/cpu_context.hpp>

using namespace mlfe;
int main(){
    TensorBlob<CPUContext> tensor_1;
    TensorBlob<CPUContext> tensor_2;
    std::vector<int> tensor_dim;
    tensor_dim.push_back(1);
    tensor_dim.push_back(3);
    tensor_dim.push_back(5);
    tensor_dim.push_back(7);
    
    /*
     * Make TensorBlob of int type.
     */
    tensor_1.Reshape<int>(tensor_dim);
    tensor_2.Reshape<int>({1, 3, 5, 7});
    
    /*
     * You can get the tensor's total elements size by calling
     * Dim(0) * Dim(1) * Dim(2) * Dim(3) or by calling Size().
     */
    int _size_tensor1 = tensor_1.Dim(0) * tensor_1.Dim(1) * tensor_1.Dim(2) * tensor_1.Dim(3);
    int _size_tensor2 = tensor_2.Size();
    /*
     * Expected result is true(1).
     */
    std::cout<<"tensor's size are equal ? "<<(_size_tensor1 == _size_tensor2)<<std::endl;
    
    /*
     * Make TensorBlob of float type.
     * If TensorBlob is not empty tensor which is allocated before,
     *  then the Reshape function free the pre-allocated storage and allocate new storage.
     */
    tensor_1.Reshape<float>({1, 3, 5, 7});
    const float *previous_mem_address = tensor_1.GetPtrConst<float>();
    
    /*
     * But, when you want to reshape TensorBlob size that is little than previous size,
     *  Reshape function dose not reallocate new memory but size will change while using the same memory.
     */
    tensor_1.Reshape<float>({1, 3, 5, 6});
    const float *new_mem_address = tensor_1.GetPtrConst<float>();
    
    /*
     * Expected result is true(1).
     */
    std::cout<<"addresses are equal ? "<<(previous_mem_address == new_mem_address)<<std::endl;
    
    /*
     * You want to totally new allocate with new size,
     * Sse Clear function and Reshape.
     */
    
    tensor_1.Clear();
    tensor_1.Reshape<float>({2, 4, 6, 8});
    
    /*
     * Set all tensor's elements by const value.
     */
    tensor_1.SetByConst<float>(3.141592);
    return 0;
}
