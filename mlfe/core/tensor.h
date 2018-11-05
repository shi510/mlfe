#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__
#include "variable.h"
#include "device.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace mlfe{
//forward declaration.
class Tensor;
class Attribution;
class OpAlgoContext;

namespace functional{

Tensor variable(std::vector<int> shape);

Tensor reshape(Tensor x, std::vector<int> shape);

void fill(Tensor to, const Tensor from);

} // end namespace functional

class Tensor final : public Variable{
public:
    struct AssignOpFunctor;

    Tensor();

    Tensor(std::string name);

    explicit Tensor(std::vector<int> shape);

    Tensor(const Tensor &t) = default;

    ~Tensor();

    bool operator==(const Tensor &v) const;

    void add_parent(const Tensor p);

    void add_child(const Tensor c);

    std::vector<Tensor> get_parents() const;

    std::vector<Tensor> get_children() const;

    int get_exec_order() const;

    void add_attr(Attribution attr);

    template <typename T>
    T get_attr(std::string name);

    OpAlgoContext get_context() const;

    memory_ptr get_memory() const;

    template <typename T>
    inline const T *host_data();

    template <typename T>
    inline T *mutable_host_data();

    template <typename T>
    inline const T *device_data();

    template <typename T>
    inline T *mutable_device_data();

    void eval();

    void backprob();

    Tensor grad();

protected:
    const void *_host_data();

    void *_mutable_host_data();

    const void *_device_data();

    void *_mutable_device_data();

    void compute_gradient(const Tensor root);

private:
    friend Tensor functional::variable(std::vector<int>);
    friend Tensor functional::reshape(Tensor x, std::vector<int> shape);
    friend struct std::hash<Tensor>;
    friend struct AssignOpFunctor;
    struct InternalData;
    std::shared_ptr<InternalData> internal_data;
};

template <typename T>
T Tensor::get_attr(std::string name){
    throw std::string("Tensor::get_attr : accessed unsupported type.");
}

template <typename T>
inline const T *Tensor::host_data(){
    return static_cast<const T *>(_host_data());
}

template <typename T>
inline T *Tensor::mutable_host_data(){
    return static_cast<T *>(_mutable_host_data());
}

template <typename T>
inline const T *Tensor::device_data(){
    return static_cast<const T *>(_device_data());
}

template <typename T>
inline T *Tensor::mutable_device_data(){
    return static_cast<T *>(_mutable_device_data());
}

template <>
bool Tensor::get_attr<bool>(std::string name);

template <>
int Tensor::get_attr<int>(std::string name);

template <>
double Tensor::get_attr<double>(std::string name);

template <>
float Tensor::get_attr<float>(std::string name);

template <>
std::vector<int> Tensor::get_attr<std::vector<int>>(std::string name);

struct Tensor::AssignOpFunctor{
    AssignOpFunctor(Tensor t, OpAlgoContext cxt);
};

} // end namespace mlfe

namespace std{

template <>
struct hash<mlfe::Tensor>{
    size_t operator()(const mlfe::Tensor &v) const;
};

} // end namespace std
#endif // end ifndef __TENSOR_HPP__
