#include "thread_pool.hpp"

namespace ocv{

namespace thread{

thread_pool::thread_pool(size_t pool_size)
    :   stop_(false)
{
    for(size_t i = 0;i < pool_size; ++i){
        workers_.emplace_back(
                    [this]()
        {
            for(;;){
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock,
                                    [this]{ return stop_ || !tasks_.empty(); });
                    if(stop_ && tasks_.empty())
                        return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        }
        );
    }
}

thread_pool::~thread_pool()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for(std::thread &worker: workers_){
        worker.join();
    }
}

}

}
