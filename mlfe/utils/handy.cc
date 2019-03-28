#include "mlfe/utils/handy.h"

namespace mlfe{
namespace util{
using riter = range_impl::iterator;

riter::iterator(int bound, int step)
    : _bound(bound), _step(step){}

riter::this_type riter::operator++(){
    _bound += _step;
    return *this;
}

riter::this_type riter::operator++(int){
    this_type prev = *this;
    _bound += _step;
    return prev;
}

riter::reference riter::operator*(){
    return _bound;
}

bool riter::operator==(const this_type& rhs) const{
    return _bound == rhs._bound;
}

bool riter::operator!=(const this_type& rhs) const{
    return _bound != rhs._bound;
}

range_impl::range_impl(int to)
    : _from(0), _to(to), _step(1) {}

range_impl::range_impl(int from, int to, int step)
    : _from(from), _to(to), _step(step) {}

riter range_impl::begin() const{
    return iterator(_from, _step);
}

riter range_impl::end() const{
    return iterator(_to, _step);
}

range_impl range(int to){
    return range_impl(to);
}

range_impl range(int from, int to){
    return range_impl(from, to);
}

range_impl range(int from, int to, int step){
    return range_impl(from, to, step);
}

} // end namespace util
} // end namespace mlfe
