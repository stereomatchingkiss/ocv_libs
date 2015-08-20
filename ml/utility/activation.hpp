#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <opencv2/core.hpp>

/*! \file activation.hpp
    \brief collection of various activation algorithm
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

struct dsigmoid
{
    void operator()(cv::Mat &inout) const
    {
        cv::multiply(1.0 - inout, inout, inout);
    }

    void operator()(cv::Mat const &input,
                    cv::Mat &output) const
    {
        cv::multiply(1.0 - input, input, output);
    }
};

cv::Mat dsigmoid_func(cv::Mat const &input)
{
    cv::Mat output;
    dsigmoid()(input, output);

    return output;
}

struct sigmoid
{
    void operator()(cv::Mat &inout) const
    {
        operator()(inout, inout);
    }

    void operator()(cv::Mat const &input,
                    cv::Mat &output) const
    {
        cv::multiply(input, -1.0, output);
        cv::exp(output, output);
        output += 1.0;
        cv::divide(1.0, output, output);
    }
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/


#endif // ACTIVATION_HPP

