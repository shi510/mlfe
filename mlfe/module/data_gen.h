#pragma once
#include <vector>
#include <type_traits>
#include <utility>
#include <iostream>

namespace mlfe{
namespace module{

template <typename Derived>
class generator
{
public:
	auto operator()(int batch_idx);

	virtual size_t size() const = 0;
};

template <typename Derived>
auto generator<Derived>::operator()(int batch_idx)
{
	return static_cast<Derived*>(this)->call(batch_idx);
}

} // end namespace module
} // end namespace mlfe