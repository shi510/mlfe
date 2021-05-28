#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__
#include "device.h"
#include "mlfe/utils/types.h"
#include "mlfe/core/graph.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace mlfe{
//forward declaration.
class Tensor;
class OpAlgoContext;

namespace functional{

Tensor create_variable(std::vector<int> shape, const bool trainable = false);

Tensor create_variable(std::vector<int> shape, type::TypeInfo ti, const bool trainable = false);

Tensor reshape(Tensor x, std::vector<int> shape);

} // end namespace functional

class Tensor final
{
public:
    template <typename T>
    struct iterator{
        using this_type = iterator;
        using value_type = T;
        using reference = T &;
        using pointer = T *;
        using difference_type = int;
        using iterator_category = std::forward_iterator_tag;

        iterator(pointer ptr) : _ptr(ptr){}

        this_type operator++(){ ++_ptr; return *this; }

        this_type operator++(int){ this_type prev = *this; ++_ptr; return prev; }

        reference operator*() const{ return *_ptr; }

        pointer operator->() const{ return _ptr; }

        bool operator==(const this_type& rhs) const{ return _ptr == rhs._ptr; }

        bool operator!=(const this_type& rhs) const{ return _ptr != rhs._ptr; }

        this_type operator+(int n){ _ptr += n; return *this; }

        this_type operator-(int n){ _ptr -= n; return *this; }

    private:
        pointer _ptr;
    };

    template <typename T>
    struct const_iterator{
        using this_type = const_iterator;
        using value_type = const T;
        using reference = const T &;
        using pointer = const T *;
        using difference_type = int;
        using iterator_category = std::forward_iterator_tag;

        const_iterator(pointer ptr) : _ptr(ptr){}

        this_type operator++(){ ++_ptr; return *this; }

        this_type operator++(int){ this_type prev = *this; ++_ptr; return prev; }

        reference operator*() const{ return *_ptr; }

        pointer operator->() const{ return _ptr; }

        bool operator==(const this_type& rhs) const{ return _ptr == rhs._ptr; }

        bool operator!=(const this_type& rhs) const{ return _ptr != rhs._ptr; }

        this_type operator+(int n){ _ptr += n; return *this; }

        this_type operator-(int n){ _ptr -= n; return *this; }

    private:
        pointer _ptr;
    };

    Tensor(const bool trainable = false);

    Tensor(std::string name, const bool trainable = false);

    explicit Tensor(std::vector<int> shape,
        const std::string name = "",
        const bool trainable = false);

    Tensor(const Tensor &t) = default;

    bool operator==(const Tensor &v) const;

    template <typename T>
    Tensor & operator=(const std::vector<T> vals){
        this->copy_from(vals);
        return *this;
    }

    Tensor weak_copy();

    void set_context(OpAlgoContext ctx);

    OpAlgoContext& get_context() const;

    memory_ptr get_memory() const;

    std::string name() const;

    void set_name(std::string name);

    void set_trainable(const bool trainable);

    void stop_gradient(bool stop_grad);

    bool trainable() const;

    void reshape(std::vector<int> shape);

    void resize(std::vector<int> shape, type::TypeInfo ti = type::float32());

    int size() const;

    int dims() const;

    int dim(int idx) const;

    const std::vector<int>& shape() const;

    type::TypeInfo type() const;

    std::shared_ptr<graph> get_graph() const;

    void set_gradient(Tensor t);

    void set_gradient_v2();

    void set_node(node n);

    node& get_node() const;

    void set_backprop_node(node n);

    node& get_backprop_node() const;

    template <typename T>
    iterator<T> begin();

    template <typename T>
    const_iterator<T> cbegin();

    template <typename T>
    iterator<T> end();

    template <typename T>
    const_iterator<T> cend();

    template <typename T>
    inline const T *data() const;

    template <typename T>
    inline T *mutable_data();

    template <typename T>
    inline const T *device_data() const;

    template <typename T>
    inline T *mutable_device_data();

    void eval();

    void backprop();

    Tensor grad();

    void zero();

    void one();

    Tensor view(std::vector<int> shape);

    template <typename T>
    void copy_from(std::vector<T> vec);

    void add_grad_marker(std::function<void (Tensor &)> marker);

    void backprop_v2();

    Tensor grad_v2() const;

    Tensor operator+=(const Tensor & other);
    Tensor operator-=(const Tensor & other);
    Tensor operator*=(const Tensor & other);
    Tensor operator/=(const Tensor & other);

    Tensor operator+(const Tensor & other) const;
    Tensor operator-(const Tensor & other) const;
    Tensor operator*(const Tensor & other) const;
    Tensor operator/(const Tensor & other) const;

    template <typename T,
        typename = std::enable_if_t<std::is_same<T, float>::value>
    >
    static Tensor from_vector(std::vector<T> vec, std::vector<int> shape){
        Tensor out = functional::create_variable(shape);
        std::copy(vec.begin(), vec.end(), out.begin<T>());
        return out;
    }

    template <typename T,
        typename = std::enable_if_t<std::is_fundamental_v<T>>
    >
    static Tensor from_scalar(const T val){
        Tensor out = functional::create_variable({});
        out.mutable_data<T>()[0] = val;
        return out;
    }

protected:
    const void *_host_data() const;

    void *_mutable_host_data();

    const void *_device_data() const;

    void *_mutable_device_data();

private:
    friend Tensor functional::create_variable(std::vector<int>, const bool trainable);
    friend Tensor functional::create_variable(std::vector<int> shape, type::TypeInfo ti, const bool trainable);
    friend Tensor functional::reshape(Tensor x, std::vector<int> shape);
    friend struct std::hash<Tensor>;

    template <typename T>
    struct weak_or_shared{
        std::weak_ptr<T> w;
        std::shared_ptr<T> s;
        T * operator->() const {
            if(s){ return s.get(); }
            return w.lock().get();
        }
        T * get() const {
            if(s){ return s.get(); }
            return w.lock().get();
        }
        std::shared_ptr<T> shared() const { return s; }
        bool is_weak() const { return s ? false : true; }
    };
    struct impl;
    weak_or_shared<impl> _pimpl;
};

Tensor operator+(const float & val, const Tensor & x);
Tensor operator-(const float & val, const Tensor & x);
Tensor operator/(const float & val, const Tensor & x);
Tensor operator*(const float & val, const Tensor & x);

Tensor operator+(const Tensor & x, const float & val);
Tensor operator-(const Tensor & x, const float & val);
Tensor operator/(const Tensor & x, const float & val);
Tensor operator*(const Tensor & x, const float & val);

template <typename T>
Tensor::iterator<T> Tensor::begin(){ return iterator<T>(mutable_data<T>()); }

template <typename T>
Tensor::const_iterator<T> Tensor::cbegin(){ return const_iterator<T>(data<T>()); }

template <typename T>
Tensor::iterator<T> Tensor::end(){ return iterator<T>(mutable_data<T>() + size()); }

template <typename T>
Tensor::const_iterator<T> Tensor::cend(){ return const_iterator<T>(data<T>() + size()); }

template <typename T>
inline const T *Tensor::data() const { return static_cast<const T *>(_host_data()); }

template <typename T>
inline T *Tensor::mutable_data(){ return static_cast<T *>(_mutable_host_data()); }

template <typename T>
inline const T *Tensor::device_data() const { return static_cast<const T *>(_device_data()); }

template <typename T>
inline T *Tensor::mutable_device_data(){ return static_cast<T *>(_mutable_device_data()); }

} // end namespace mlfe

namespace std{

template <>
struct hash<mlfe::Tensor>{
    size_t operator()(const mlfe::Tensor &v) const;
};

} // end namespace std
#endif // end ifndef __TENSOR_HPP__
