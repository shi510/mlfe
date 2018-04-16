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
    tensor_1.Resize<int>(tensor_dim);
    tensor_2.Resize<int>({1, 3, 5, 7});
    
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
     *  then the Resize function free the pre-allocated storage and allocate new storage.
     */
    tensor_1.Resize<float>({1, 3, 5, 7});
    const float *previous_mem_address = tensor_1.GetPtrConst<float>();
    
    /*
     * But, when you want to reshape TensorBlob size that is little than previous size,
     *  Resize function dose not reallocate new memory but size will change while using the same memory.
     */
    tensor_1.Resize<float>({1, 3, 5, 6});
    const float *new_mem_address = tensor_1.GetPtrConst<float>();
    
    /*
     * Expected result is true(1).
     */
    std::cout<<"addresses are equal ? "<<(previous_mem_address == new_mem_address)<<std::endl;
    
    /*
     * By using Clear and Resize function,
     * you can totally new allocate with new size.
     */
    
    tensor_1.Clear();
    tensor_1.Resize<float>({2, 4, 6, 8});
    
    /*
     * Or you can copy from anothor tensor.
     * The operator = not allocate new memory,
     * just copy the memory address.
     */
    tensor_1.Clear();
    tensor_1 = tensor_2;
    
    /*
     * You can change the tensor dimention shape.
     * The shape size must be same with previous size.
     * In this case, it changes from 4 dim (1, 3, 5, 7) to 2 dim (2, 105).
     * It holds the data and memory, not re-allocate the memory.
     * If the size dose not match, it will throw an error.
     */
    tensor_1.Reshape({1, 3 * 5 * 7});
    
    /*
     * You can check the tensor's type.
     */
    std::cout<<"Does tensor_1 has int type? "<<tensor_1.MatchType<int>()<<std::endl;
    std::cout<<"Does tensor_1 has float type? "<<tensor_1.MatchType<float>()<<std::endl;
    return 0;
}
