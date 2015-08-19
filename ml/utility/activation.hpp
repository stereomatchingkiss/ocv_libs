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

struct sigmoid
{
    void operator()(cv::Mat &inout) const
    {
        inout *= -1.0;
        cv::exp(inout, inout);
        inout += 1.0;
        //inout = 1.0 / inout;
        cv::divide(1.0, inout, inout);
    }
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/


#endif // ACTIVATION_HPP

