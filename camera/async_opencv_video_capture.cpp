#include "async_opencv_video_capture.hpp"

using namespace cv;

namespace ocv{

namespace camera{

async_opencv_video_capture::async_opencv_video_capture(std::function<bool (const std::exception &)> exception_listener) :
    cam_exception_listener_(std::move(exception_listener)),
    stop_(false)
{
}

void async_opencv_video_capture::add_listener(std::function<void (cv::Mat)> listener, void *key)
{
    listeners_.insert(std::make_pair(key, std::move(listener)));
}

bool async_opencv_video_capture::is_stop() const noexcept
{
    return stop_;
}

bool async_opencv_video_capture::open_url(const std::string &url)
{
    return cap_.open(url);
}

void async_opencv_video_capture::run()
{
    std::thread([this]()
    {
        for(Mat frame; stop_ == false;){
            try{
                cap_>>frame;
                for(auto &val : listeners_){
                    val.second(frame);
                }
            }catch(std::exception const &ex){
                cam_exception_listener_(ex);
            }
        }
    }).detach();
}

void async_opencv_video_capture::set_video_capture(VideoCapture cap)
{
    cap_ = cap;
}

void async_opencv_video_capture::stop()
{
    stop_ = true;
}

}

}
