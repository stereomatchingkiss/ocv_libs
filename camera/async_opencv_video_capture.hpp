#ifndef OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
#define OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace ocv{

namespace camera{

/**
 * Read the video asynchronous
 *
 * @code
 * #include <ocv_libs/camera/async_opencv_video_capture.hpp>
 * #include <opencv2/core.hpp>
 * #include <opencv2/highgui.hpp>
 *
 * #include <iostream>
 * #include <mutex>
 *
 * int main(int argc, char *argv[])
 * {
 *     if(argc != 2){
 *         std::cerr<<"must enter url of media\n";
 *         return -1;
 *     }
 *
 *     std::mutex emutex;
 *     //create the functor to handle the exception when cv::VideoCapture fail
 *     //to capture the frame and wait 30 msec between each frame
 *     long long constexpr wait_msec = 30;
 *     ocv::camera::async_opencv_video_capture<> cl([&](std::exception const &ex)
 *     {
 *         //cerr of c++ is not a thread safe class, so we need to lock the mutex
 *         std::unique_lock<std::mutex> lock(emutex);
 *         std::cerr<<"camera exception:"<<ex.what()<<std::endl;
 *         return true;
 *     }, wait_msec);
 *     cl.open_url(argv[1]);
 *
 *     //add listener to process captured frame
 *     cv::Mat img;
 *     cl.add_listener([&](cv::Mat input)
 *     {
 *         std::unique_lock<std::mutex> lock(emutex);
 *         img = input;
 *     }, &emutex);
 *
 *     //execute the task(s)
 *     cl.run();
 *
 *     //We must display the captured image at main thread but not
 *     //in the listener, because every manipulation related to gui
 *     //must perform in the main thread(it also called gui thread)
 *     for(int finished = false; finished != 'q';){
 *         finished = std::tolower(cv::waitKey(30));
 *         std::unique_lock<std::mutex> lock(emutex);
 *         if(!img.empty()){
 *             cv::imshow("frame", img);
 *         }
 *     }
 * }
 * @endcode
 */
template<typename Mutex = std::mutex>
class async_opencv_video_capture
{
public:
    using listener_key = void const*;

    /**
     * @param cam_exception_listener a listener to process exception if the video capture throw exception
     * @param wait_msec Determine how many milliseconds the videoCapture will wait before capture next frame
     * @param replay Restart the camera if the frame is empty
     * @warning If mutex type is std::mutex, unless the listener do not run in the same thread of videoCapture,
     * else do not call the api of async_opencv_video_capture in the listener, this may cause dead lock. Use
     * std::recursive_mutex if you want to do that
     */
    explicit async_opencv_video_capture(std::function<bool(std::exception const &ex)> cam_exception_listener,
                                        long long wait_msec = 0,
                                        bool replay = true);
    ~async_opencv_video_capture();

    /**
     * Add listener to process frame captured by the videoCapture
     * @warning
     * 1. Must handle exception in the listener
     * 2. If mutex type is std::mutex, unless the listener do not run in the same thread of videoCapture,
     * else do not call the api of async_opencv_video_capture in the listener, this may cause dead lock. Use
     * std::recursive_mutex if you want to do that
     */
    void add_listener(std::function<void(cv::Mat)> listener, listener_key key)
    {
        std::lock_guard<Mutex> lock(mutex_);
        listeners_.emplace_back(key, std::move(listener));
    }
    bool is_stop() const noexcept
    {
        std::lock_guard<Mutex> lock(mutex_);
        return stop_;
    }
    bool open_url(std::string const &url)
    {
        std::lock_guard<Mutex> lock(mutex_);
        url_ = url;
        return cap_.open(url);
    }

    void remove_listener(listener_key key);
    /**
     * The thread will start and detach after you call this function.
     * @warning
     * 1. add listener(s) before you call this function
     * 2. add video capture before you call this function
     */
    void run()
    {
        if(thread_){
            set_stop(true);
            thread_->join();
            set_stop(false);
        }

        create_thread();
    }
    void set_video_capture(cv::VideoCapture cap)
    {
        std::lock_guard<Mutex> lock(mutex_);
        cap_ = cap;
    }
    /**
     *Determine how many milliseconds the videoCapture will wait before
     *capture next frame, default value is 0
     */
    void set_wait(long long msec)
    {
        std::lock_guard<Mutex> lock(mutex_);
        wait_for_ = std::chrono::milliseconds(msec);
    }
    void stop()
    {
        set_stop(true);
    }

private:
    using listener_pair = std::pair<listener_key, std::function<void(cv::Mat)>>;
    using listeners_vec = std::vector<listener_pair>;

    void create_thread()
    {
        thread_ = std::make_unique<std::thread>([this]()
        {
            //read the frames in infinite for loop
            for(;;){
                cv::Mat frame;
                std::lock_guard<Mutex> lock(mutex_);
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
    void set_stop(bool val)
    {
        std::lock_guard<Mutex> lock(mutex_);
        stop_ = val;
    }

    std::function<bool(std::exception const &ex)> cam_exception_listener_;
    cv::VideoCapture cap_;
    listeners_vec listeners_;
    mutable Mutex mutex_;
    bool replay_;
    bool stop_;
    std::unique_ptr<std::thread> thread_;
    std::string url_;
    std::chrono::milliseconds wait_for_;
};

template<typename Mutex>
async_opencv_video_capture<Mutex>::
async_opencv_video_capture(std::function<bool (const std::exception &)> cam_exception_listener,
                           long long wait_msec,
                           bool replay) :
    cam_exception_listener_(std::move(cam_exception_listener)),
    replay_(replay),
    stop_(false),
    wait_for_(std::chrono::milliseconds(wait_msec))
{
}

template<typename Mutex>
async_opencv_video_capture<Mutex>::~async_opencv_video_capture()
{
    stop();
    thread_->join();
}

template<typename Mutex>
void async_opencv_video_capture<Mutex>::remove_listener(async_opencv_video_capture::listener_key key)
{
    std::lock_guard<Mutex> lock(mutex_);
    auto it = std::find_if(std::begin(listeners_), std::end(listeners_), [&](auto const &val)
    {
        return val.first == key;
    });
    if(it != std::end(listeners_)){
        listeners_.erase(it);
    }
}

}

}

#endif // OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
