#include "evaluator.h"
#include "graph.h"

namespace mlfe{

Evaluator::Evaluator(Device device) : d(device){}

void Evaluator::Init(optimizer::AppliedOptimizer *ao){
    auto target_name = ao->Target().Name() + "ForOpt";
    if(target_algos.count(target_name) <= 0){
        std::vector<OpAlgoPtr> new_task;
        std::vector<OpDependency> deps;
        auto v_list = visit_bfs(ao->Target());
        std::reverse(v_list.begin(), v_list.end());
        for(auto &t : v_list){
            if(t.get_dep().Name() != "Unknown"){
                deps.push_back(t.get_dep());
            }
        }
        AllocateAllVars(deps);
        AllocateAllVars(ao->get_bwd_op_deps());
        auto model_algos = FindOpAlgo(deps);
        auto model_grad_algos = FindOpAlgo(ao->get_bwd_op_deps());
        auto opt_algos = FindOpAlgo(ao->get_update_op_deps());

        new_task.insert(
            new_task.end(),
            model_algos.begin(),
            model_algos.end()
        );
        new_task.insert(
            new_task.end(),
            model_grad_algos.begin(),
            model_grad_algos.end()
        );
        new_task.insert(
            new_task.end(),
            opt_algos.begin(),
            opt_algos.end()
        );
        target_algos[target_name] = new_task;
        InitAllVariable(deps);
    }
}

void Evaluator::Init(std::vector<Tensor> ts){
    for(auto t : ts){
        if(target_algos.count(t.Name()) <= 0){
            std::vector<OpAlgoPtr> new_task;
            std::vector<OpDependency> deps;
            auto v_list = visit_bfs(t);
            std::reverse(v_list.begin(), v_list.end());
            for(auto &t : v_list){
                if(t.get_dep().Name() != "Unknown"){
                    deps.push_back(t.get_dep());
                }
            }
            AllocateAllVars(deps);
            auto model_algos = FindOpAlgo(deps);
            new_task.insert(
                new_task.end(),
                model_algos.begin(),
                model_algos.end()
            );
            target_algos[t.Name()] = new_task;
            InitAllVariable(deps);
        }
    }
}

void Evaluator::AllocateAllVars(std::vector<OpDependency> deps){
    for(auto op_dep : deps){
        for(auto in : op_dep.Inputs()){
            if(tensor_refs.count(in.Name()) <= 0){
                auto ref = std::make_shared<TensorMemRef>(in, d.CreateDeviceMemory());
                tensor_refs[in.Name()] = ref;
            }
        }

        for(auto out : op_dep.Outputs()){
            if(tensor_refs.count(out.Name()) <= 0){
                auto ref = std::make_shared<TensorMemRef>(out, d.CreateDeviceMemory());
                tensor_refs[out.Name()] = ref;
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
            auto init_dep = in.InitDependency();
            if(init_dep.Name() != "Unknown" &&
                already_init.count(init_dep.UniqueName()) <= 0){
                OpAlgoContext oac(d, &tensor_refs, init_dep.Context());
                std::stringstream ss;
                ss << "Name:" << init_dep.Name();
                if(d.Name() == Device::CUDA::string_cudnn ||
                    d.Name() == Device::CUDA::string){
                    ss << "/Device:" << Device::CUDA::string;
                }
                else{
                    ss << "/Device:" << Device::CPU::string;
                }
                oar->GetOpAlgo(ss.str(), &oac)->Compute();
                already_init[init_dep.UniqueName()] = in;
            }
        }
    }
}
} // end namespace mlfe;
