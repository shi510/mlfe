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

Tensor create_variable(std::vector<int> shape);

Tensor reshape(Tensor x, std::vector<int> shape);

void fill(Tensor to, const Tensor from);

} // end namespace functional

class Tensor final : public Variable{
public:
    template <typename T>
    class iterator;

    template <typename T>
    class const_iterator;

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
    iterator<T> begin();

    template <typename T>
    const_iterator<T> cbegin();

    template <typename T>
    iterator<T> end();

    template <typename T>
    const_iterator<T> cend();

    template <typename T>
    inline const T *data();

    template <typename T>
    inline T *mutable_data();

    template <typename T>
    inline const T *device_data();

    template <typename T>
    inline T *mutable_device_data();

    void eval();

    void backprop();

    Tensor grad();

protected:
    const void *_host_data();

    void *_mutable_host_data();

    const void *_device_data();

    void *_mutable_device_data();

    void compute_gradient(const Tensor root);

private:
    friend Tensor functional::create_variable(std::vector<int>);
    friend Tensor functional::reshape(Tensor x, std::vector<int> shape);
    friend struct std::hash<Tensor>;
    friend struct AssignOpFunctor;
    struct impl;
    std::shared_ptr<impl> _pimpl;
};

template <typename T>
T Tensor::get_attr(std::string name){
    throw std::string("Tensor::get_attr : accessed unsupported type.");
}

template <typename T>
Tensor::iterator<T> Tensor::begin(){
    return iterator<T>(mutable_data<T>());
}

template <typename T>
Tensor::const_iterator<T> Tensor::cbegin(){
    return const_iterator<T>(data<T>());
}

template <typename T>
Tensor::iterator<T> Tensor::end(){
    return iterator<T>(mutable_data<T>() + size());
}

template <typename T>
Tensor::const_iterator<T> Tensor::cend(){
    return const_iterator<T>(data<T>() + size());
}

template <typename T>
inline const T *Tensor::data(){
    return static_cast<const T *>(_host_data());
}

template <typename T>
inline T *Tensor::mutable_data(){
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

template <typename T>
class Tensor::iterator{
public:
    using this_type = iterator;
    using value_type = T;
    using reference = T &;
    using pointer = T *;
    using difference_type = int;
    using iterator_category = std::forward_iterator_tag;

    iterator(pointer ptr) : _ptr(ptr){}

    this_type operator++(){
        ++_ptr;
        return *this;
    }

    this_type operator++(int){
        this_type prev = *this;
        ++_ptr;
        return prev;
    }

    reference operator*() const{
        return *_ptr;
    }

    pointer operator->() const{
        return _ptr;
    }

    bool operator==(const this_type& rhs) const{
        return _ptr == rhs._ptr;
    }

    bool operator!=(const this_type& rhs) const{
        return _ptr != rhs._ptr;
    }

private:
    pointer _ptr;
};

template <typename T>
class Tensor::const_iterator{
public:
    using this_type = const_iterator;
    using value_type = const T;
    using reference = const T &;
    using pointer = const T *;
    using difference_type = int;
    using iterator_category = std::forward_iterator_tag;

    const_iterator(pointer ptr) : _ptr(ptr){}

    this_type operator++(){
        ++_ptr;
        return *this;
    }

    this_type operator++(int){
        this_type prev = *this;
        ++_ptr;
        return prev;
    }

    reference operator*() const{
        return *_ptr;
    }

    pointer operator->() const{
        return _ptr;
    }

    bool operator==(const this_type& rhs) const{
        return _ptr == rhs._ptr;
    }

    bool operator!=(const this_type& rhs) const{
        return _ptr != rhs._ptr;
    }

private:
    pointer _ptr;
};

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
