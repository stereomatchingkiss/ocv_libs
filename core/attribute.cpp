#include "attribute.hpp"

#include <boost/format.hpp>

#include <iostream>
#include <sstream>

namespace ocv{

contour_attribute::
contour_attribute(std::vector<cv::Point> const &contour, double epsillon):
    area_{cv::contourArea(contour)},
    bounding_rect_{cv::boundingRect(contour)},
    aspect_ratio_{bounding_rect_.width / bounding_rect_.height},
    perimeter_{cv::arcLength(contour, true)}
{
    std::vector<cv::Point> buffer;
    cv::convexHull(contour, buffer);
    solidity_ = area_/cv::contourArea(buffer);

    cv::approxPolyDP(contour, buffer, perimeter_ * epsillon, true);
    poly_size_ = buffer.size();
}

void print_contour_attribute(std::vector<cv::Point> const &contour,
                             double epsillon,
                             std::ostream &out)
{    
    contour_attribute const ca(contour, epsillon);

    std::ostringstream ostr;
    ostr<<boost::format("%=11f|%=11f|%=11f|%=11f|"
                        "%=11f|%=11f")
          %ca.area_%ca.perimeter_%ca.aspect_ratio_
          %ca.extent_%ca.solidity_%ca.poly_size_;
    out<<ostr.str()<<std::endl;
}

void print_contour_attribute_name(std::ostream &out)
{
    std::ostringstream ostr;
    ostr<<boost::format("%=11s|%=11s|%=11s|%=11s|%=11s|%=11s")
          % "Area" % "Perimeter" % "Aspect"
          % "Extent" % "Solidity" % "PolySize";
    out<<ostr.str()<<std::endl;
}

}
