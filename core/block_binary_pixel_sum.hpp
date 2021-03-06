#ifndef BLOCK_BINARY_SUM_H
#define BLOCK_BINARY_SUM_H

#include "for_each.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * Implementation of block binary pixel sum
 *
 */
template<typename T = float>
class block_binary_pixel_sum
{
public:
    explicit block_binary_pixel_sum(cv::Size const &target_size = {30,15},
                                    std::vector<cv::Size> block_sizes =
    {{5,5}, {5,10}, {10,5}, {10,10}});

    std::vector<T> const& describe(cv::Mat const &input);

    std::vector<cv::Size> const& get_block_sizes() const
    {
        return block_sizes_;
    }

    cv::Size const get_target_size() const
    {
        return target_size();
    }

private:
    std::vector<cv::Size> block_sizes_;
    std::vector<T> features_;
    cv::Mat gray_mat_;    
    cv::Size target_size_;
};

/**
 * Initialize target size and block size
 * @param target_size Size of the input image will resize to
 * if needed
 * @param block_size The set of M x N regions that the
 * descriptor will use when computing the ratio of foreground
 * pixels to total pixels in each block.
 */
template<typename T>
block_binary_pixel_sum<T>::
block_binary_pixel_sum(cv::Size const &target_size,
                       std::vector<cv::Size> block_sizes) :
    block_sizes_(std::move(block_sizes)),
    target_size_(target_size)
{

}

/**
 * Extract block binary pixel sum features from input
 * @param input Input image, the range of the input should
 * within [0,255]
 * @return Features of block binary pixel sum
 */
template<typename T>
std::vector<T> const&
block_binary_pixel_sum<T>::describe(const cv::Mat &input)
{
    CV_Assert(input.channels() == 1);

    cv::resize(input, gray_mat_, target_size_);

    features_.clear();
    auto func = [this](int, int, cv::Mat const &data)
    {
        features_.emplace_back(cv::countNonZero(data) /
                               static_cast<T>(target_size_.area()));
    };
    for(auto const &bsize : block_sizes_){
        ocv::for_each_block(gray_mat_, bsize,
                            func, bsize);
    }

    return features_;
}

} /*! @} End of Doxygen Groups*/

#endif // BLOCK_BINARY_SUM_H
