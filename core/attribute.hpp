#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * Analyze attributes of contour
 */
struct contour_attribute
{
    /**
     * Generate contour attributes
     * @param contour the contour want to measure the attributes
     * @param epsillon epsillon value to approximate the contour
     */
    contour_attribute(std::vector<cv::Point> const &contour,
                      double epsillon);

    double counter_area_;
    cv::Rect bounding_rect_;
    double bounding_area_;
    double aspect_ratio_;
    double perimeter_;
    double extent_;
    double solidity_;
    size_t poly_size_;

private:
    std::vector<cv::Point> buffer_;
};

/**
 * print the attribute data of contour, could be slow and
 * fat, only use it when develop
 * @param contour the contour want to measure the attributes
 * @param epsillon epsillon value to approximate the contour
 * (use by 3rd argument of cv::approxPolyDP())
 * @param out the attributes will be print by this ostream
 */
void print_contour_attribute(std::vector<cv::Point> const &contour,
                             double epsillon,
                             std::ostream &out);

/**
 * Print the name of th contour attribute, expected to be called
 * before the function "print_contour_attribute" by once.
 * @param out the attributes name will be print by this ostream
 */
void print_contour_attribute_name(std::ostream &out);

} /*! @} End of Doxygen Groups*/

#endif // ATTRIBUTE_HPP

