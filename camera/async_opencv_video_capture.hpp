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
 *     ocv::camera::async_opencv_video_capture cl([&](std::exception const &ex)
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
class async_opencv_video_capture
{
public:
    using listener_key = void const*;

    /**
     * @param cam_exception_listener a listener to process exception if the video capture throw exception
     * @param wait_msec Determine how many milliseconds the videoCapture will wait before capture next frame
     * @param replay Restart the camera if the frame is empty
     * @warning Unless the listener do not run in the same thread of videoCapture, else do not call the api
     * of async_opencv_video_capture in the listener, this may cause dead lock
     */
    explicit async_opencv_video_capture(std::function<bool(std::exception const &ex)> cam_exception_listener,
                                        long long wait_msec = 0,
                                        bool replay = true);
    ~async_opencv_video_capture();

    /**
     * Add listener to process frame captured by the videoCapture
     * @warning
     * 1. Must handle exception in the listener
     * 2. Unless the listener do not run in the same thread of videoCapture, else do not call the api
     * of async_opencv_video_capture in the listener, this may cause dead lock
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
    bool replay_;
    bool stop_;
    std::unique_ptr<std::thread> thread_;
    std::string url_;
    std::chrono::milliseconds wait_for_;
};

}

}

#endif // OCV_CAMERA_ASYNC_OPENCV_VIDEO_CAPTURE_HPP
