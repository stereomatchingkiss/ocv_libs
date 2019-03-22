#ifndef OCV_THREAD_THREAD_POOL_HPP
#define OCV_THREAD_THREAD_POOL_HPP

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

namespace ocv{

namespace thread{

/**
 * This project alter from https://github.com/progschj/ThreadPool
 */
class thread_pool {
public:
    explicit thread_pool(size_t pool_size = 1);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)->std::future<typename std::result_of<F(Args...)>::type>;
    ~thread_pool();

private:
    std::condition_variable condition_;
    // synchronization
    std::mutex queue_mutex_;
    bool stop_;
    // the task queue
    std::queue<std::function<void()>> tasks_;
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers_;
};

// the constructor just launches some amount of workers


// add new work item to the pool
template<class F, class... Args>
auto thread_pool::enqueue(F&& f, Args&&... args)->std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if(!stop_){
            tasks_.emplace([task](){ (*task)(); });
        }
    }
    condition_.notify_one();

    return res;
}

}

}

// the destructor joins all threads


#endif // OCV_THREAD_THREAD_POOL_HPP
