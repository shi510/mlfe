#include "context.hpp"

namespace mlfe {

DEFINE_REGISTRY(
    ContextSwitchCopyRegistry,
    std::string,
    std::shared_ptr<ContextSwitchCopier>
)

DEFINE_REGISTRY(
    ContextRegistry,
    std::string,
    std::shared_ptr<Context>
)

Context::Context(Accelerator acc) 
    : acc_str(to_string(acc)){ }

Context::~Context() {}

std::shared_ptr<Context> Context::Create(Accelerator acc) {
    if (acc == Accelerator::CUDNN) {
        acc = Accelerator::CUDA;
    }
    return ContextRegistry()->Create("Context_" + to_string(acc));
}

void Context::Allocate(const int size, const int block_size) {
    try {
        Allocator(size, block_size);
    }
    catch (std::string &e) {
        throw e;
    }
}

void Context::Copy(
    const std::shared_ptr<Context> src, 
    std::shared_ptr<Context> dst
){
    std::string functor_str = "Context_Copy_";
    functor_str += src->acc_str + "_";
    functor_str += dst->acc_str;
    auto f = ContextSwitchCopyRegistry()->Create(functor_str);
    f->copy(src, dst);
}

} /* namespace mlfe */
