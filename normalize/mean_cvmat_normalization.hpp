#ifndef OCV_NORMALIZE_MEAN_CV_IMAGE_NORMALIZATION_HPP
#define OCV_NORMALIZE_MEAN_CV_IMAGE_NORMALIZATION_HPP

#include <opencv2/core.hpp>

#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup normalize
 *  @{
 */
namespace normalize{

/**
 *transform cv::Mat to zero mean matrix
 */
class mean_cv_image_normalization
{
public:
    explicit mean_cv_image_normalization(int result_type) :
        result_type_(result_type) {}

    void fit(std::vector<cv::Mat> const &input)
    {
        if(!input.empty()){
            if(!input[0].empty()){
                mean_.create(input[0].rows, input[0].cols, result_type_);
                mean_.setTo(0);
            }
            find_mean(input);
        }
    }

    std::vector<cv::Mat>
    fit_transform(std::vector<cv::Mat> const &input)
    {
        std::vector<cv::Mat> output;
        fit_transform(input, output);

        return output;
    }

    void fit_transform(std::vector<cv::Mat> const &input,
                       std::vector<cv::Mat> &output)
    {
        fit(input);
        transform(input, output);
    }

    cv::Mat transform(cv::Mat const &input)
    {
        cv::Mat output;
        transform(input, output);

        return output;
    }
    void transform(cv::Mat const &input, cv::Mat &output)
    {
        input.convertTo(output, result_type_);
        output -= mean_;
    }
    std::vector<cv::Mat> transform(std::vector<cv::Mat> const &input)
    {
        std::vector<cv::Mat> output;
        transform(input, output);

        return output;
    }
    void transform(std::vector<cv::Mat> const &input,
                   std::vector<cv::Mat> &output)
    {
        output.resize(input.size());
        for(size_t i = 0; i != input.size(); ++i){
            transform(input[i], output[i]);
        }
    }

private:
    void find_mean(std::vector<cv::Mat> const &input)
    {
        for(auto const &mat : input){
            mean_ += mat;
        }
        mean_ /= static_cast<double>(input.size());
    }

    cv::Mat mean_;
    int result_type_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif //OCV_NORMALIZE_MEAN_CV_IMAGE_NORMALIZATION_HPP
