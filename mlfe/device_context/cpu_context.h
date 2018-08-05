#ifndef __CPU_CONTEXT_HPP__
#define __CPU_CONTEXT_HPP__
#include <random>
#include "context.h"

namespace mlfe {
    
class CPUContext final : public Context {
public:
    ~CPUContext() override;

    static std::mt19937 rng;
};
    
} // end namespace mlfe
#endif // end #ifndef __CPU_CONTEXT_HPP__
