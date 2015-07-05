#include "histogram.hpp"

namespace ocv
{

/**
 * @brief encapsulation of cv::caclHist, all of the meaning of the parameters are same as\n
 * the cv::caclHist.\n
 * ex : cv::Mat hist = calc_histogram({hsv}, {0, 1}, {180, 256}, {{ {0, 180}, {0, 256} }});
 * @param images input images list
 * @param output  Output histogram, which is a dense or sparse\n
 * dimensional(equal to channels.size()) array.
 * @param channels List of the dims channels used to\n
 * compute the histogram. The first array channels\n
 * are numerated from 0 to images[0].channels()-1,\n
 * the second array channels are counted from\n
 * images[0].channels() to images[0].channels() +\n
 * images[1].channels()-1, and so on.
 * @param hist_sizes Array of histogram sizes in\n
 * each dimension.
 * @param ranges Array of the dims arrays of the histogram bin\n
 * boundaries in each dimension
 * @param mask Optional mask. If the matrix is not empty,\n
 * it must be an 8-bit array of the same size as images[i].\n
 * The non-zero mask elements mark the array elements\n
 * counted in the histogram.
 * @param uniform  Flag indicating whether the histogram\n
 * is uniform or not
 * @param accumulate Accumulation flag. If it is set, the\n
 * histogram is not cleared in the beginning when it is\n
 * allocated. This feature enables you to compute a\n
 * single histogram from several sets of arrays, or to\n
 * update the histogram in time.
 */
void calc_histogram(std::initializer_list<cv::Mat> images,
                    cv::OutputArray output,
                    std::initializer_list<int> channels,
                    std::initializer_list<int> hist_sizes,
                    std::initializer_list<float[2]> ranges,
cv::InputArray mask,
bool uniform,
bool accumulate)
{
    size_t const sizes = ranges.size();
    std::vector<float const*> d_ranges(sizes);
    for(size_t i = 0; i != sizes; ++i){
        d_ranges[i] = *(std::begin(ranges) + i);
    }

    cv::calcHist(std::begin(images), images.size(), std::begin(channels), mask, output,
                 channels.size(), std::begin(hist_sizes), &d_ranges[0], uniform ,accumulate);
}

/**
 * @brief rgb_histogram calculate the histograms of rgb,\n
 * the meaning of the parameters are same as calc_histogram
 */
void rgb_histogram(const cv::Mat &input, cv::Mat &output,
                   std::initializer_list<int> hist_sizes,
                   std::initializer_list<float[2]> ranges,
                   cv::InputArray mask)
{
    if(input.channels() < 3){
        throw std::underflow_error("input.channels() < 3");
    }
    calc_histogram<3>({input}, output, {0, 1, 2}, hist_sizes,
    ranges, mask);
}

/**
 * @brief rgb_histogram overload of rgb_histogram
 * @return rgb histogram
 */
cv::Mat rgb_histogram(const cv::Mat &input,
                      std::initializer_list<int> hist_sizes,
                      std::initializer_list<float[2]> ranges,
                      cv::InputArray mask)
{
    cv::Mat output;
    rgb_histogram(input, output, hist_sizes, ranges, mask);

    return output;
}

}
