#include "evaluator.h"

namespace mlfe{

Evaluator::Evaluator(Device device) : d(device){}

void Evaluator::Init(optimizer::AppliedOptimizer *ao){
    auto target_name = ao->InputGrad().Name();
    if(target_algos.count(target_name) <= 0){
        std::vector<OpAlgoPtr> new_task;
        AllocateAllVars(ao->InputGrad().OpDependencies());
        auto model_algos = FindOpAlgo(ao->InputGrad().OpDependencies());
        auto opt_algos = FindOpAlgo(ao->OpDependencies());

        new_task.insert(
            new_task.end(),
            model_algos.begin(),
            model_algos.end()
        );
        new_task.insert(
            new_task.end(),
            opt_algos.begin(),
            opt_algos.end()
        );
        target_algos[target_name] = new_task;
        InitAllVariable(ao->Target().OpDependencies());
    }
}

void Evaluator::Init(std::vector<Tensor> ts){
    for(auto t : ts){
        if(target_algos.count(t.Name()) <= 0){
            std::vector<OpAlgoPtr> new_task;
            AllocateAllVars(t.OpDependencies());
            AllocateAllVars({ t.InitDependency() });
            auto model_algos = FindOpAlgo(t.OpDependencies());
            new_task.insert(
                new_task.end(),
                model_algos.begin(),
                model_algos.end()
            );
            target_algos[t.Name()] = new_task;
            InitAllVariable(t.OpDependencies());
            InitAllVariable({ t.InitDependency() });
        }
    }
}

void Evaluator::AllocateAllVars(std::vector<OpDependency> deps){
    for(auto op_dep : deps){
        for(auto in : op_dep.Inputs()){
            auto var = std::get<1>(in);
            if(tensor_refs.count(var.Name()) <= 0){
                auto ref = std::make_shared<TensorMemRef>(var, d.CreateDeviceMemory());
                tensor_refs[var.Name()] = ref;
            }
        }

        for(auto out : op_dep.Outputs()){
            auto var = std::get<1>(out);
            if(tensor_refs.count(var.Name()) <= 0){
                auto ref = std::make_shared<TensorMemRef>(var, d.CreateDeviceMemory());
                tensor_refs[var.Name()] = ref;
            }
        }
    }
}

std::vector<Evaluator::OpAlgoPtr>
Evaluator::FindOpAlgo(std::vector<OpDependency> deps){
    auto oar = OpAlgoRegistry::Get();
    std::vector<OpAlgoPtr> algos;

    for(auto dep : deps){
        if(global_op_algos.count(dep.UniqueName()) <= 0){
            OpAlgoContext oac(d, &tensor_refs, dep.Context());
            std::stringstream ss;
            ss << "Name:" << dep.Name();
            for(auto in : dep.Inputs()){
                ss << "/In:" << std::get<1>(in).Type().type;
            }

            for(auto out : dep.Outputs()){
                ss << "/Out:" << std::get<1>(out).Type().type;
            }
            if(d.Name() != Device::CPU::string){
                auto try_cudnn = ss.str() + "/Device:" + Device::CUDA::string_cudnn;
                if(Device::CUDA::enable_cudnn){
                    if(!oar->Has(try_cudnn)){
                        try_cudnn = ss.str() + "/Device:" + Device::CUDA::string;
                    }
                }
                else{
                    try_cudnn = ss.str() + "/Device:" + Device::CUDA::string;
                }
                global_op_algos[dep.UniqueName()] = oar->GetOpAlgo(try_cudnn, &oac);
            }
            else{
                ss << "/Device:" << Device::CPU::string;
                global_op_algos[dep.UniqueName()] = oar->GetOpAlgo(ss.str(), &oac);
            }
        }
        algos.push_back(global_op_algos[dep.UniqueName()]);
    }
    return algos;
}

void Evaluator::InitAllVariable(std::vector<OpDependency> deps){
    auto oar = OpAlgoRegistry::Get();

    for(auto dep : deps){
        for(auto in : dep.Inputs()){
            auto init_dep = std::get<1>(in).InitDependency();
            if(init_dep.Name() != "Unknown" &&
                already_init.count(init_dep.UniqueName()) <= 0){
                OpAlgoContext oac(d, &tensor_refs, init_dep.Context());
                std::stringstream ss;
                ss << "Name:" << init_dep.Name();
                for(auto in : init_dep.Inputs()){
                    ss << "/In:" << std::get<1>(in).Type().type;
                }
                for(auto out : init_dep.Outputs()){
                    ss << "/Out:" << std::get<1>(out).Type().type;
                }
                if(d.Name() == Device::CUDA::string_cudnn ||
                    d.Name() == Device::CUDA::string){
                    ss << "/Device:" << Device::CUDA::string;
                }
                else{
                    ss << "/Device:" << Device::CPU::string;
                }
                oar->GetOpAlgo(ss.str(), &oac)->Compute();
                already_init[init_dep.UniqueName()] = std::get<1>(in);
            }
        }
    }
}
} // end namespace mlfe;
