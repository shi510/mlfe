#include "tensor.h"
#include "../utils/assert.h"
#include "op_dep.h"
#include "op_design.h"

namespace mlfe{

struct Tensor::InternalData{
    InternalData()
        : init_dep(OpDependency::Builder("Unknown").Finish()){}
    std::vector<OpDependency> op_deps;
    OpDependency init_dep;
};

using TDA = Tensor::DependencyAdder;
using TIDA = Tensor::InitDependencyAdder;

TIDA::InitDependencyAdder(OpDependency dep)
    : target_dep(&dep){}

TDA::DependencyAdder(OpDependency dep)
    : target_dep(&dep){
    ExtractDepFromInputs();
}

void TDA::ExtractDepFromInputs(){
    auto inputs = target_dep->Inputs();
    for(auto in : inputs){
        auto in_deps = std::get<1>(in).OpDependencies();
        auto unique_deps = RemoveDuplicatedDep(in_deps);
        all_deps.insert(
            all_deps.end(),
            unique_deps.begin(),
            unique_deps.end()
        );
    }
}

std::vector<OpDependency>
TDA::RemoveDuplicatedDep(std::vector<OpDependency> &deps){
    std::vector<OpDependency> unique_dep;
    for(auto dep : deps){
        if(!IsIn(dep)){
            unique_dep.push_back(dep);
        }
    }

    return unique_dep;
}

bool TDA::IsIn(const OpDependency &target){
    auto target_name = target.UniqueName();
    for(auto dep : all_deps){
        auto dep_name = dep.UniqueName();
        if(dep_name == target_name){
            return true;
        }
    }
    return false;
}

Tensor &Tensor::operator=(DependencyAdder dep_adder){
    internal_data->op_deps.insert(
        internal_data->op_deps.end(),
        dep_adder.all_deps.begin(),
        dep_adder.all_deps.end()
    );
    internal_data->op_deps.push_back(*dep_adder.target_dep);
    return *this;
}

Tensor &Tensor::operator=(InitDependencyAdder dep_adder){
    internal_data->init_dep = *dep_adder.target_dep;
    return *this;
}

Tensor::Tensor()
    : internal_data(std::make_shared<InternalData>()){}

Tensor::Tensor(std::string name)
    : Variable(name),
    internal_data(std::make_shared<InternalData>()){}

Tensor::Tensor(std::vector<int> shape)
    : Variable(shape),
    internal_data(std::make_shared<InternalData>()){}

Tensor::~Tensor(){}

std::string Variable::Name() const{
    return *_name + "_" + std::to_string(_id->Id());
}

// TODO : This code is a temporary solution for
//         initialization of tensor values.
//        I'll update this code to make awesome.
void Tensor::Initialize(Tensor init){
    *this = init;
    internal_data->init_dep = internal_data->op_deps[0];
    internal_data->op_deps.clear();
}

std::vector<OpDependency> Tensor::OpDependencies() const{
    return internal_data->op_deps;
}

OpDependency Tensor::InitDependency() const{
    return internal_data->init_dep;
}

unsigned int Variable::UniqueID::_next_gen = 0;

} // end namespace mlfe
