#ifndef OCV_SALIENCY_PYRAMID_SCAN_HPP
#define OCV_SALIENCY_PYRAMID_SCAN_HPP

#include "../core/for_each.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup saliency
 *  @{
 */
namespace saliency{

/**
 *@Scan the image by image pyramids.
 */
class pyramid_scan
{
public:
    /**
     * @param min_size minimum size of the pyramid image,
     * the image will stop to scale down if width or height
     * <= min_size
     * @param win_size Size of the sliding windows, should not
     * larger than min_size
     * @param step The pixels going to skip in both the (x,y)
     * direction
     * @param scale Small scale yiels more layers in the pyramid,
     * larger scale yield less layers.Should bigger than one
     */
    pyramid_scan(cv::Size2i const &min_size,
                 cv::Size2i const &win_size,
                 cv::Size2i const &step,
                 double scale);

    template<typename WinFunc>
    void scan(cv::Mat const &input, WinFunc win_func)
    {
        cv::Mat img;
        input.copyTo(img);
        for(auto cur_size = input.size();
            cur_size.height >= min_size_.height &&
            cur_size.width >= min_size_.width;
            cur_size = img.size()){

            double const wratio = input.cols / static_cast<double>(cur_size.width);
            double const hratio = input.rows / static_cast<double>(cur_size.height);
            ocv::for_each_block(img, win_size_,
                                win_func, step_,
                                wratio, hratio);

            cv::resize(img, img,
                       cv::Size(img.cols / scale_, img.rows / scale_));            
        }
    }

private:
    cv::Size2i min_size_;
    double scale_;
    cv::Size2i step_;
    cv::Size2i win_size_;
};

pyramid_scan::pyramid_scan(cv::Size2i const &min_size,
                              cv::Size2i const &win_size,
                              cv::Size2i const &step,
                              double scale) :
    min_size_(min_size),
    scale_(scale),
    step_(step),
    win_size_(win_size)
{
    CV_Assert(min_size_.width > 0 && min_size_.height > 0);
    CV_Assert(scale_ > 1.0);
    CV_Assert(step_.width > 0 && step_.height > 0);
    CV_Assert(win_size_.width > 0 && win_size_.height > 0);
    CV_Assert(win_size_.width <= min_size_.width &&
              win_size_.height <= min_size_.height);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_SALIENCY
