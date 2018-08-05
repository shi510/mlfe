#include <string>
#include <new>
#include <functional>
#include "cpu_context.h"

namespace mlfe {

std::mt19937 CPUContext::rng = std::mt19937(1357);

CPUContext::~CPUContext(){}

} /* namespace mlfe */
