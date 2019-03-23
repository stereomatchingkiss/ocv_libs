#ifndef OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
#define OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace ocv{

namespace camera{

/**
 * Read the video asynchronous
 * @code
 *
 * @endcode
 */
class async_opencv_video_capture
{
public:
    using listener_key = void const*;

    /**
     * @param listener a listener to process exception if the video capture throw exception
     */
    explicit async_opencv_video_capture(std::function<bool(std::exception const &ex)> exception_listener);
    ~async_opencv_video_capture();

    /**
     * Add listener to process frame captured by the videoCapture
     * @warning Must handle exception in the listener
     */
    void add_listener(std::function<void(cv::Mat)> listener, listener_key key);
    bool is_stop() const noexcept;
    bool open_url(std::string const &url);

    void remove_listener(listener_key key);
    /**
     * The thread will start and detach after you call this function.
     * @warning
     * 1. add listener(s) before you call this function
     * 2. add video capture before you call this function
     */
    void run();
    void set_video_capture(cv::VideoCapture cap);
    void stop();

private:
    using listener_pair = std::pair<listener_key, std::function<void(cv::Mat)>>;
    using listeners_vec = std::vector<listener_pair>;

    std::function<bool(std::exception const &ex)> cam_exception_listener_;
    cv::VideoCapture cap_;
    listeners_vec listeners_;
    std::mutex mutex_;
    std::atomic<bool> stop_;
    std::unique_ptr<std::thread> thread_;
    void create_thread();
};

}

}

#endif // OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
