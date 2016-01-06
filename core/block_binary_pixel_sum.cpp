#include "block_binary_pixel_sum.hpp"

#include "for_each.hpp"

#include <opencv2/imgproc.hpp>

namespace ocv{

/**
 * Initialize target size and block size
 * @param target_size Size of the input image will resize to
 * if needed
 * @param block_size The set of M x N regions that the
 * descriptor will use when computing the ratio of foreground
 * pixels to total pixels in each block.
 */
block_binary_pixel_sum::
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
std::vector<double> const&
block_binary_pixel_sum::describe(const cv::Mat &input)
{    
    cv::resize(input, resize_mat_, target_size_);
    if(resize_mat_.type() != CV_8U){
        resize_mat_.convertTo(gray_mat_, CV_8U);
    }else{
        gray_mat_ = resize_mat_;
    }

    features_.clear();
    auto func = [this](int, int, cv::Mat const &data)
    {
        features_.emplace_back(cv::countNonZero(data) /
                               static_cast<double>(target_size_.area()));
    };
    for(auto const &bsize : block_sizes_){
        ocv::for_each_block(gray_mat_, bsize,
                            func, bsize);
    }

    return features_;
}


}
