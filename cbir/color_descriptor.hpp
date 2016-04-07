#ifndef OCV_CBIR_COLOR_DESCRIPTOR_HPP
#define OCV_CBIR_COLOR_DESCRIPTOR_HPP

#include <opencv2/core.hpp>

#include <array>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup cbir
 *  @{
 */
namespace cbir{

/**
 * Encapsulate the cv::calcHist api, make it easier to extract
 * the histogram from image
 * ocv::cbir::color_descriptor cd;
 * cv::cvtColor(img, img, CV_BGR2HSV);
 * @code
 * cd.set_channels({0,1});
 * cd.set_dim(2);
 * cd.set_hist_size({30,32});
 * cd.set_ranges({{0,180}, {0,256}});
 * cd.get_descriptor(img);
 * @endcode
 */
class color_descriptor
{
public:    
    explicit color_descriptor();

    void get_descriptor(cv::Mat const &input,
                        cv::Mat &output,
                        cv::Mat const &mask = cv::Mat());
    cv::Mat get_descriptor(cv::Mat const &input,
                           cv::Mat const &mask = cv::Mat());

    void set_accumulate(bool accu);
    void set_channels(std::vector<int> const &channels);
    void set_dim(int dim);
    void set_hist_size(std::vector<int> const &hist_size);
    void set_normalize(bool norm);
    void set_ranges(std::vector<std::array<float,2>> const &ranges);
    void set_uniform(bool uniform);

private:
    bool accu_;
    std::vector<int> channels_;
    int dim_;
    std::vector<float const*> hist_range_;
    std::vector<int> hist_size_;
    bool norm_;
    std::vector<cv::Mat> planes_;
    std::vector<std::array<float,2>> ranges_;
    bool uniform_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_COLOR_DESCRIPTOR_HPP
