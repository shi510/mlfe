#ifndef __THREAD_POOL_HPP__
#define __THREAD_POOL_HPP__
#include <string>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include "assert.hpp"

namespace mlfe{
class ThreadPool{
public:
    explicit ThreadPool(unsigned int size){
        runtime_assert(size == 1, "thread size must be 1.(in current version, supported upto 1)");
        is_stop = false;
        for(int n = 0; n < size; ++n){
            threads.push_back(std::thread(std::bind(&ThreadPool::InternalExecutor, this)));
        }
    }
    
    ~ThreadPool(){
        is_stop = true;
        cv.notify_all();
        for(int n = 0; n < threads.size(); ++n){
            threads[n].join();
        }
    }
    
    void AddTask(const std::function<void ()> &task, int id){
        std::unique_lock<std::mutex> lock(m);
        tasks.push(std::make_pair(task, id));
        task_state[id] = false;
        cv.notify_one();
    }
    
    void Wait(int id){
        std::unique_lock<std::mutex> lock(m);
        while(!task_state[id]){
            state_noti.wait(lock);
        }
    }
    
    bool IsFinished(int id){
        return task_state[id];
    }
    
private:
    void InternalExecutor(){
        while(true){
            std::unique_lock<std::mutex> lock(m);
            while(tasks.empty() && !is_stop){
                cv.wait(lock);
            }
            if(is_stop){
                break;
            }
            auto task = tasks.front();
            tasks.pop();
            lock.unlock();
            task.first();
            lock.lock();
            task_state[task.second] = true;
            state_noti.notify_one();
        }
    }
    std::condition_variable cv;
    std::condition_variable state_noti;
    std::mutex m;
    std::queue<std::pair<std::function<void ()>, int>> tasks;
    std::vector<std::thread> threads;
    std::map<int, bool> task_state;
    bool is_stop;
};
} /* namespace mlfe */
#endif /* __THREAD_POOL_HPP__ */
