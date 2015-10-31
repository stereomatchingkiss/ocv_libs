#ifndef BITPLANE_HPP
#define BITPLANE_HPP

#include <opencv2/core.hpp>

#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

void bitplane_generator(cv::Mat const &input, std::vector<cv::Mat> &output);

} /*! @} End of Doxygen Groups*/

#endif // BITPLANE_HPP

