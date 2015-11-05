#ifndef OCV_RESIZE_HPP
#define OCV_RESIZE_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * Resize the image with keeping the aspect ratio of it
 * @param input input image
 * @param output input image after resize
 * @param size If the width do not equal to zero, this function will resize
 * the width to desire target and keep the aspect ratio of height and vice versa
 * @param interpolation interpoloation methods support by cv::resize
 * @code
 * cv::Mat input = cv::imread("ninokuni.png"); //size is 1156 * 780
 * cv::Mat output;
 * //output size will become 480 * 323
 * cv::resize_aspect_ratio(input, output, {480,0});
 * //output size will become 474 * 320
 * cv::resize_aspect_ratio(input, output, {0, 320});
 * @endcode
 */
void resize_aspect_ratio(cv::Mat const &input, cv::Mat &output, cv::Size const &size,
                         int interpolation = cv::INTER_LINEAR);
	
} /*! @} End of Doxygen Groups*/

#endif
