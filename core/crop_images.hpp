#ifndef OCV_CORE_CROP_IMAGES_HPP
#define OCV_CORE_CROP_IMAGES_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * Crop the image based on the center
 * @param input input image
 * @param x_center horizontal center of image
 * @param y_center vertical center of image
 * @param x_extend pixels extend from x_center
 * @param y_extend pixels extend from y_center
 * @return image after cropped, this image is the subview of
 * the input, if the input dead, this image will become invalid
 * @code
 * auto img = cv::imread("m7.jpg");
 * auto crop_img = crop_image(img, 16, 34, 32, 32);
 * @endcode
 */
cv::Mat crop_image(cv::Mat const &input,
                   int x_center, int y_center,
                   int x_extend, int y_extend);

cv::Mat crop_image(cv::Mat const &input,
                   float x_center, float y_center,
                   int x_extend, int y_extend);

std::vector<cv::Mat> crop_image(std::string const &directory,
                                std::string const &img_name,
                                cv::Size2i const &size = {16, 16});

std::vector<cv::Mat> crop_directory_images(std::string const &directory,
                                           cv::Size2i const &size = {16, 16});

} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_CROP_IMAGES_HPP

