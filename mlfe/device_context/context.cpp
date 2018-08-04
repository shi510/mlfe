#include "context.hpp"

namespace mlfe {

Context::Context(Accelerator acc)
    : acc_str(to_string(acc)){ }

Context::~Context() {}

void Context::Allocate(const int size, const int block_size) {
    try {
        Allocator(size, block_size);
    }
    catch (std::string &e) {
        throw e;
    }
}

} /* namespace mlfe */
