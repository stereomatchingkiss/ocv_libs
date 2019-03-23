#include "async_opencv_video_capture.hpp"

using namespace cv;
using namespace std;

namespace ocv{

namespace camera{

async_opencv_video_capture::
async_opencv_video_capture(std::function<bool (const std::exception &)> exception_listener,
                           long long wait_msec,
                           bool replay) :
    cam_exception_listener_(std::move(exception_listener)),
    replay_(replay),
    stop_(false),
    wait_for_(chrono::milliseconds(wait_msec))
{
}

async_opencv_video_capture::~async_opencv_video_capture()
{
    stop();
    thread_->join();
}

void async_opencv_video_capture::add_listener(std::function<void (cv::Mat)> listener, listener_key key)
{
    unique_lock<mutex> lock(mutex_);
    listeners_.emplace_back(key, std::move(listener));
}

bool async_opencv_video_capture::is_stop() const noexcept
{
    unique_lock<mutex> lock(mutex_);
    return stop_;
}

bool async_opencv_video_capture::open_url(const std::string &url)
{
    unique_lock<mutex> lock(mutex_);
    url_ = url;
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
        listeners_.erase(it);
    }
}

void async_opencv_video_capture::create_thread()
{
    thread_ = std::make_unique<std::thread>([this]()
    {
        //read the frames in infinite for loop
        for(Mat frame;;){
            unique_lock<mutex> lock(mutex_);
            if(!stop_ && !listeners_.empty()){
                try{
                    cap_>>frame;
                }catch(std::exception const &ex){
                    //reopen the camera if exception thrown ,this may happen frequently when you
                    //receive frames from network
                    cap_.open(url_);
                    cam_exception_listener_(ex);
                }

                if(!frame.empty()){
                    for(auto &val : listeners_){
                        val.second(frame);
                    }
                }else{
                    if(replay_){
                        cap_.open(url_);
                    }else{
                        break;
                    }
                }
                std::this_thread::sleep_for(wait_for_);
            }else{
                break;
            }
        }
    });
}

void async_opencv_video_capture::set_stop(bool val)
{
    unique_lock<mutex> lock(mutex_);
    stop_ = val;
}

void async_opencv_video_capture::run()
{    
    if(thread_){
        set_stop(true);
        thread_->join();
        set_stop(false);
    }

    create_thread();
}

void async_opencv_video_capture::set_video_capture(VideoCapture cap)
{
    unique_lock<mutex> lock(mutex_);
    cap_ = cap;
}

void async_opencv_video_capture::set_wait(long long msec)
{
    unique_lock<mutex> lock(mutex_);
    wait_for_ = chrono::milliseconds(msec);
}

void async_opencv_video_capture::stop()
{
    set_stop(true);
}

}

}
