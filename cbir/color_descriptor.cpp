#include "color_descriptor.hpp"

#include <opencv2/imgproc.hpp>

namespace ocv{

namespace cbir{

color_descriptor::color_descriptor() :
    accu_{false},    
    dim_{1},
    uniform_{true}
{

}

void color_descriptor::
get_descriptor(const cv::Mat &input,
               std::vector<cv::Mat> &output,
               const cv::Mat &mask)
{    
    cv::split(input, planes_);
    constexpr int nimages = 1;    
    cv::calcHist(&planes_[0], nimages, &channels_[0],
            mask, output, dim_,
            &hist_size_[0], &hist_range_[0], uniform_, accu_);//*/
}

std::vector<cv::Mat>
color_descriptor::get_descriptor(const cv::Mat &input,
                                 const cv::Mat &mask)
{
    std::vector<cv::Mat> output;
    get_descriptor(input, output, mask);

    return output;
}

void color_descriptor::set_accumulate(bool accu)
{
    accu_ = accu;
}

void color_descriptor::set_channels(const std::vector<int> &channels)
{
    channels_ = channels;
}

void color_descriptor::set_dim(int dim)
{
    dim_ = dim;
}

void color_descriptor::set_hist_size(const std::vector<int> &hist_size)
{
    hist_size_ = hist_size;
}

void color_descriptor::set_uniform(bool uniform)
{
    uniform_ = uniform;
}

void color_descriptor::
set_ranges(const std::vector<std::array<float, 2>> &ranges)
{
    ranges_ = ranges;
    hist_range_.clear();
    for(auto const &range : ranges_){
        hist_range_.emplace_back(&range[0]);
    }
}

}

}
