#ifndef FEATURE_SCALING_HPP
#define FEATURE_SCALING_HPP

#include <opencv2/core.hpp>

/*! \file gradient_checking.hpp
    \brief check the results of gradient descent like algorithm
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup ml
 *  @{
 */
namespace ml{

enum class sample_type{
    row,
    col
};

void zero_mean(cv::Mat const &input,
               cv::Mat &output,
               sample_type type = sample_type::col);

void z_score_scaling(cv::Mat const &input,
                     cv::Mat &output);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // FEATURE_SCALING_HPP

