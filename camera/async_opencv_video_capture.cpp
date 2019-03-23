#include "async_opencv_video_capture.hpp"

using namespace cv;
using namespace std;

namespace ocv{

namespace camera{

async_opencv_video_capture::async_opencv_video_capture(std::function<bool (const std::exception &)> exception_listener) :
    cam_exception_listener_(std::move(exception_listener)),
    stop_(false)
{
}

async_opencv_video_capture::~async_opencv_video_capture()
{
    thread_->join();
}

void async_opencv_video_capture::add_listener(std::function<void (cv::Mat)> listener, listener_key key)
{
    unique_lock<mutex> lock(mutex_);
    listeners_.emplace_back(key, std::move(listener));
}

bool async_opencv_video_capture::is_stop() const noexcept
{
    return stop_;
}

bool async_opencv_video_capture::open_url(const std::string &url)
{
    return cap_.open(url);
}

void async_opencv_video_capture::remove_listener(listener_key key)
{
    unique_lock<mutex> lock(mutex_);
    auto it = std::find_if(std::begin(listeners_), std::end(listeners_), [&](auto const &val)
    {
        return val.first == key;
    });
    if(it != std::end(listeners_)){
        listeners_.erase(listeners_.begin());
    }
}

void async_opencv_video_capture::create_thread()
{
    thread_ = std::make_unique<std::thread>([this]()
    {
        for(Mat frame; stop_ == false;){
            unique_lock<mutex> lock(mutex_);
            if(!listeners_.empty()){
                try{
                    cap_>>frame;
                    if(!frame.empty()){
                        for(auto &val : listeners_){
                            val.second(frame);
                        }
                    }
                }catch(std::exception const &ex){
                    cam_exception_listener_(ex);
                }
            }else{
                stop_ = true;
            }
        }
    });
}

void async_opencv_video_capture::run()
{
    stop_ = false;
    if(thread_){
        thread_->join();
    }

    create_thread();
}

void async_opencv_video_capture::set_video_capture(VideoCapture cap)
{
    unique_lock<mutex> lock(mutex_);
    cap_ = cap;
}

void async_opencv_video_capture::stop()
{
    stop_ = true;
}

}

}
