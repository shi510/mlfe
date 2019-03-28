#ifndef __HANDY_H__
#define __HANDY_H__
#include <iterator>

namespace mlfe{
namespace util{

class range_impl final{
public:
    class iterator;

    iterator begin() const;

    iterator end() const;

private:
    range_impl(int to);

    range_impl(int from, int to, int step = 1);

    int _from;
    int _to;
    int _step;

    friend range_impl range(int);
    friend range_impl range(int, int);
    friend range_impl range(int, int, int);
};

class range_impl::iterator final{
public:
    using this_type = iterator;
    using value_type = int;
    using reference = value_type &;
    using pointer = int *;
    using difference_type = int;
    using iterator_category = std::forward_iterator_tag;

    this_type operator++();

    this_type operator++(int);

    reference operator*();

    bool operator==(const this_type& rhs) const;

    bool operator!=(const this_type& rhs) const;

private:
    iterator(int bound, int step);

    int _bound;
    int _step;

    friend class range_impl;
};

range_impl range(int to);

range_impl range(int from, int to);

range_impl range(int from, int to, int step);

} // end namespace util
} // end namespace mlfe
#endif // __HANDY_H__
