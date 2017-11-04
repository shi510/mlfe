#ifndef __ASSERT_HPP__
#define __ASSERT_HPP__
#include <string>

namespace mlfe{

inline void runtime_assert(bool condition, std::string message){
    !condition ? throw message : 0;
}

} /* namespace mlfe */
#endif /* __ASSERT_HPP__ */
