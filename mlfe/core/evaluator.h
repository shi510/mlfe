#ifndef __EVALUATOR_HPP__
#define __EVALUATOR_HPP__
#include "op_algo.h"
#include "op_dep.h"
#include "device.h"
#include "tensor.h"
#include "tensor_mem_ref.h"
#include "../optimizers/optimizer.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <sstream>

namespace mlfe{
class OpAlgo;
class OpDependency;

class Evaluator{
using OpAlgoPtr = std::shared_ptr<OpAlgo>;
public:
    template <class T>
    using Feeder = std::vector<std::tuple<Tensor, std::vector<T>>>;

    template <class Dev>
    static Evaluator Create();

    template <class T>
    void Run(optimizer::AppliedOptimizer ao, Feeder<T> feed_input);

    template <class T>
    std::vector<T> Run(Tensor fetcher, Feeder<T> feed_input);

    template <class T>
    std::vector<std::vector<T>> Run(std::vector<Tensor> fetchers, Feeder<T> feed_input);

    template <class T>
    std::vector<T> Run(Tensor fetcher);

    template <class T>
    std::vector<std::vector<T>> Run(std::vector<Tensor> fetchers);

protected:
    Evaluator(Device d);

    void Init(optimizer::AppliedOptimizer *ao);

    void Init(std::vector<Tensor> ts);

    void AllocateAllVars(std::vector<OpDependency> deps);

    std::vector<OpAlgoPtr> FindOpAlgo(std::vector<OpDependency> deps);

    void InitAllVariable(std::vector<OpDependency> deps);

private:
    Tensor target;
    Device d;
    std::map<std::string, OpAlgoPtr> global_op_algos;
    std::map<std::string, std::shared_ptr<TensorMemRef>> tensor_refs;
    std::map<std::string, std::vector<OpAlgoPtr>> target_algos;
    std::map<std::string, Tensor> already_init;
    std::function<void(DeviceMemory, DeviceMemory)> in_copy;
    std::function<void(DeviceMemory, DeviceMemory)> out_copy;
};

template <class Dev>
Evaluator Evaluator::Create(){
    Evaluator eval = Device(Device::Select<Dev>());
    eval.in_copy = Device::Copy<Device::CPU, Dev>;
    eval.out_copy = Device::Copy<Dev, Device::CPU>;
    return eval;
}

template <class T>
void Evaluator::Run(optimizer::AppliedOptimizer ao,
                    Feeder<T> feed_input
                   )
{
    Init(&ao);
    auto &runner = target_algos[ao.InputGrad().Name()];
    for(auto feed : feed_input){
        auto ref = tensor_refs[std::get<0>(feed).Name()];
        auto from = reinterpret_cast<type::uint8::T *>(std::get<1>(feed).data());
        if(ref->Size() != std::get<1>(feed).size()){
            throw "The Size of " + std::get<0>(feed).Name() + " is not same with Feeder.";
        }
        DeviceMemory from_dev = Device::Select<Device::CPU>().CreateDeviceMemory();
        from_dev.Allocate(from, ref->Size() * sizeof(T));
        in_copy(
            from_dev,
            ref->GetDeviceMemory()
        );
    }

    for(auto algo : runner){
        algo->Compute();
    }
}

template <class T>
std::vector<T> Evaluator::Run(Tensor fetcher,
                              Feeder<T> feed_input
                             )
{
    return Run(std::vector<Tensor>{ fetcher }, feed_input)[0];
}

template <class T> std::vector<std::vector<T>>
    Evaluator::Run(std::vector<Tensor> fetchers,
                   Feeder<T> feed_input
                  )
{
    std::vector<std::vector<T>> ret;

    Init(fetchers);
    for(auto feed : feed_input){
        auto ref = tensor_refs[std::get<0>(feed).Name()];
        auto from = reinterpret_cast<type::uint8::T *>(std::get<1>(feed).data());
        if(ref->Size() != std::get<1>(feed).size()){
            throw "The Size of " + std::get<0>(feed).Name() + " is not same with Feeder.";
        }
        DeviceMemory from_dev = Device::Select<Device::CPU>().CreateDeviceMemory();
        from_dev.Allocate(from, ref->Size() * sizeof(T));
        in_copy(
            from_dev,
            ref->GetDeviceMemory()
        );
    }

    for(auto fetcher : fetchers){
        auto &runner = target_algos[fetcher.Name()];
        for(auto algo : runner){
            algo->Compute();
        }
    }

    for(auto t : fetchers){
        auto ref = tensor_refs[t.Name()];
        std::vector<T> host(ref->Size());
        auto to = reinterpret_cast<type::uint8::T *>(host.data());
        DeviceMemory to_dev = Device::Select<Device::CPU>().CreateDeviceMemory();
        to_dev.Allocate(to, ref->Size() * sizeof(T));
        out_copy(
            ref->GetDeviceMemory(),
            to_dev
        );
        ret.push_back(host);
    }

    return ret;
}

template <class T>
std::vector<T> Evaluator::Run(
    Tensor fetcher
){
    return Run<T>(std::vector<Tensor>{ fetcher })[0];
}

template <class T> std::vector<std::vector<T>>
    Evaluator::Run(std::vector<Tensor> fetchers)
{
    std::vector<std::vector<T>> ret;

    Init(fetchers);

    for(auto fetcher : fetchers){
        auto &runner = target_algos[fetcher.Name()];
        for(auto algo : runner){
            algo->Compute();
        }
    }

    for(auto t : fetchers){
        auto ref = tensor_refs[t.Name()];
        std::vector<T> host(ref->Size());
        auto to = reinterpret_cast<type::uint8::T *>(host.data());
        DeviceMemory to_dev = Device::Select<Device::CPU>().CreateDeviceMemory();
        to_dev.Allocate(to, ref->Size() * sizeof(T));
        out_copy(
            ref->GetDeviceMemory(),
            to_dev
        );
        ret.push_back(host);
    }

    return ret;
}

} // end namespace mlfe
#endif // end #ifndef __EVALUATOR_HPP__
