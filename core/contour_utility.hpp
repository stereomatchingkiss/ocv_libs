#ifndef OCV_CORE_CONTOUR_UTILITY_HPP
#define OCV_CORE_CONTOUR_UTILITY_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

void contours_to_bounding_rect(std::vector<std::vector<cv::Point>> const &contours,
                               std::vector<cv::Rect> &rects);

std::vector<cv::Rect> contours_to_bounding_rect(std::vector<std::vector<cv::Point>> const &contours);


void draw_convex(std::vector<std::vector<cv::Point>> const &contours, cv::Mat &img,
                 cv::Scalar const &color = {255,0,0});
void draw_rect(std::vector<std::vector<cv::Point>> const &contours, cv::Mat &img,
               cv::Scalar const &color = {255,0,0});

std::vector<std::vector<cv::Point>> find_contours(cv::Mat const &img,
                                                  int mode = cv::RETR_EXTERNAL,
                                                  int method = cv::CHAIN_APPROX_SIMPLE);

void find_contours(cv::Mat const &img, std::vector<std::vector<cv::Point>> &contours,
                   cv::Range const &range,
                   int mode = cv::RETR_EXTERNAL,
                   int method = cv::CHAIN_APPROX_SIMPLE);

std::vector<std::vector<cv::Point>> find_contours(cv::Mat const &img, cv::Range const &range,
                                                  int mode = cv::RETR_EXTERNAL,
                                                  int method = cv::CHAIN_APPROX_SIMPLE);
} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_CONTOUR_UTILITY_HPP
