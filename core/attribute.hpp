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
 * Store the attributes of contour
 */
struct contour_attribute
{   
    double counter_area_ = 0;
    cv::Rect bounding_rect_ = {};
    double bounding_area_ = 0;
    double aspect_ratio_ = 0;
    double perimeter_ = 0;
    double extent_ = 0;
    double solidity_ = 0;
    size_t poly_size_ = 0;
};

/**
 * Analyze attributes of contour
 */
struct contour_analyzer
{
public:
    /**
     * Analyze attributes of contour
     * @param contour contour want to do attribute analyze
     * @param epsillon epsillon value to approximate the contour
     * (use by 3rd argument of cv::approxPolyDP())
     * @return attribute of contour
     */
    contour_attribute analyze(std::vector<cv::Point> const &contour,
                              double epsillon);

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

