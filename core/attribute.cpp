#include "attribute.hpp"

#include <boost/format.hpp>

#include <iostream>
#include <sstream>

namespace ocv{

void print_contour_attribute(std::vector<cv::Point> const &contour,
                             double epsillon,
                             std::ostream &out)
{    
    contour_attribute const ca = contour_analyzer().analyze(contour, epsillon);
    std::ostringstream ostr;
    ostr<<boost::format("%=11.2f|%=11.2f|%=11.2f|%=11.2f|"
                        "%=11.2f|%=11.2f|%=11.2f")
          %ca.counter_area_%ca.bounding_area_%ca.perimeter_
          %ca.aspect_ratio_%ca.extent_%ca.solidity_
          %ca.poly_size_;
    out<<ostr.str()<<std::endl;
}

void print_contour_attribute_name(std::ostream &out)
{
    std::ostringstream ostr;
    ostr<<boost::format("%=11s|%=11s|%=11s|%=11s|"
                        "%=11s|%=11s|%=11s")
          % "CArea" % "BArea" %"Perimeter" % "Aspect"
          % "Extent" % "Solidity" % "PolySize";
    out<<ostr.str()<<std::endl;
}

contour_attribute
contour_analyzer::analyze(std::vector<cv::Point> const &contour,
                          double epsillon)
{
    contour_attribute ca;

    ca.counter_area_ = cv::contourArea(contour);
    ca.bounding_rect_ = cv::boundingRect(contour);
    ca.bounding_area_ = static_cast<double>(ca.bounding_rect_.area());
    ca.aspect_ratio_ = ca.bounding_rect_.width /
            static_cast<double>(ca.bounding_rect_.height);
    ca.perimeter_ = cv::arcLength(contour, true);
    ca.extent_ = ca.counter_area_/ca.bounding_area_;

    cv::convexHull(contour, buffer_);
    ca.solidity_ = ca.counter_area_/cv::contourArea(buffer_);

    cv::approxPolyDP(contour, buffer_, ca.perimeter_ * epsillon, true);
    ca.poly_size_ = buffer_.size();

    return ca;
}

}
