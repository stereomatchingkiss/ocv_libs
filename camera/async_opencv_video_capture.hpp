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
 * @code
 * if(argc != 2){
 *       std::cerr<<"must enter url of media\n";
 *       return -1;
 * }
 *
 * std::mutex emutex;
 * ocv::camera::async_opencv_video_capture cl([&](std::exception const &ex)
 * {
 *     std::unique_lock<std::mutex> lock(emutex);
 *     std::cerr<<"camera exception:"<<ex.what()<<std::endl;
 *     return true;
 * });
 * cl.open_url(argv[1]);
 *
 * cv::Mat img;
 * static size_t index = 0;
 * cl.add_listener([&](cv::Mat input)
 * {
 *     std::unique_lock<std::mutex> lock(emutex);
 *     std::cout<<"index:"<<index++<<std::endl;
 *     img = input;
 * }, &emutex);
 * cl.run();
 *
 * for(int finished = false; finished != 'q';){
 *     finished = std::tolower(cv::waitKey(30));
 *     std::unique_lock<std::mutex> lock(emutex);
 *     if(!img.empty()){
 *         cv::imshow("frame", img);
 *     }
 * }
 *
 * cl.stop();
 * @endcode
 */
class async_opencv_video_capture
{
public:
    using listener_key = void const*;

    /**
     * @param listener a listener to process exception if the video capture throw exception
     * @param wait_msec Determine how many milliseconds the videoCapture will wait before capture next frame
     */
    explicit async_opencv_video_capture(std::function<bool(std::exception const &ex)> exception_listener,
                                        long long wait_msec = 0);
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
    /**
     *Determine how many milliseconds the videoCapture will wait before
     *capture next frame, default value is 0
     */
    void set_wait(long long msec);
    void stop();

private:
    using listener_pair = std::pair<listener_key, std::function<void(cv::Mat)>>;
    using listeners_vec = std::vector<listener_pair>;

    void create_thread();
    void set_stop(bool val);

    std::function<bool(std::exception const &ex)> cam_exception_listener_;
    cv::VideoCapture cap_;
    listeners_vec listeners_;
    mutable std::mutex mutex_;
    bool stop_;
    std::unique_ptr<std::thread> thread_;
    std::chrono::milliseconds wait_for_;
};

}

}

#endif // OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
