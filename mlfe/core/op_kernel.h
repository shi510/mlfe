#pragma once
#include "mlfe/core/tuple_iter.h"
#include "mlfe/core/tensor.h"
#include <tuple>
#include <functional>
#include <string>

namespace mlfe{
namespace operators_v2{

template <typename op_type, typename fn_type>
struct op_kernel_impl
{
    template <typename ...arguments>
    auto operator()(arguments ...args)
    {
        return fn(args...);
    }

    static fn_type fn;
    static std::string name;
};

#define DECLARE_OP_KERNEL(op_name, fn_type)            \
    struct op_name##_kernel                            \
        : op_kernel_impl<op_name##_kernel, fn_type> {}

#define REGIST_OP_KERNEL(op_name, fn_type, impl_fn)                        \
    template <>                                                            \
    fn_type op_kernel_impl<op_name##_kernel, fn_type>::fn = impl_fn;       \
    template <>                                                            \
    std::string op_kernel_impl<op_name##_kernel, fn_type>::name = #op_name


template<std::size_t I = 0, typename ...A, typename ...B>
inline typename std::enable_if<I == sizeof...(B), void>::type
for_each_zip(std::tuple<A...>& a, std::tuple<B...>& b){}

template<std::size_t I = 0, typename ...A, typename ...B>
inline typename std::enable_if<I < sizeof...(B), void>::type
for_each_zip(std::tuple<A...>& a, std::tuple<B...>& b)
{
    std::get<0>(a).add_grad_marker(std::get<I>(b));
    for_each_zip<I + 1>(a, b);
}

struct marker {
    template <typename ...T>
    struct I {
        I(T... t) : list(t...){}
        std::tuple<T...> list;
    };

    template <typename ...T>
    struct O {
        O(T... t) : list(t...){
            static_assert(sizeof...(T) == 1,
                "# of kernel output should be 1.");
        }

        template <typename ...GM>
        O & operator()(GM ...grad_markers){
            auto gm_tuple = std::tuple<GM...>(grad_markers...);
            for_each_zip(list, gm_tuple);
            return *this;
        }

        std::tuple<T...> list;
    };
};

template <typename K, typename ...I, typename ...O, typename ...T>
void call(marker::I<I...> inputs, marker::O<O...> outputs, T ...args)
{
    auto xs = to_array(inputs.list);
    auto ys = to_array(outputs.list);
    for(auto y : ys){
        for(auto x : xs){
            y.get_node().add_input_v2(x.get_node());
            y.get_node().add_attr("op_name", K::name);
        }
    }
    std::apply(K::fn,
        std::tuple_cat(inputs.list, outputs.list, std::tuple<T...>(args...)));
}

} // namespace operators_v2
} // namespace mlfe
