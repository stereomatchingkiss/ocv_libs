#ifndef CROP_IMAGES_HPP
#define CROP_IMAGES_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

std::vector<cv::Mat> crop_image(std::string const &directory,
                                std::string const &img_name,
                                cv::Size2i const &size = {16, 16});

std::vector<cv::Mat> crop_directory_images(std::string const &directory,
                                           cv::Size2i const &size = {16, 16});

} /*! @} End of Doxygen Groups*/

#endif // CROP_IMAGES_HPP

