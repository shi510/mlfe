#ifndef __CONTEXT_HPP__
#define __CONTEXT_HPP__
#include <memory>
#include <type_traits>
#include <string>
#include "../utils/types.h"

namespace mlfe {

class Context { 
public:
    virtual ~Context();
};

} // end namespace mlfe
#endif // end #ifndef __CONTEXT_HPP__
