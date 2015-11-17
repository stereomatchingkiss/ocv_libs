#include "attribute.hpp"

#include <boost/format.hpp>

#include <iostream>
#include <sstream>

namespace ocv{

std::ostream& operator<<(std::ostream &out, contour_attribute const &attr)
{
    std::ostringstream ostr;
    ostr<<boost::format("%=11.2f|%=11.2f|%=11.2f|%=11.2f|"
                        "%=11.2f|%=11.2f|%=11.2f")
          %attr.contour_area_%attr.bounding_area_%attr.perimeter_
          %attr.aspect_ratio_%attr.extent_%attr.solidity_
          %attr.poly_size_;
    out<<ostr.str()<<std::endl;

    return out;
}

void print_contour_attribute(std::vector<cv::Point> const &contour,
                             double epsillon,
                             std::ostream &out)
{    
    contour_attribute const ca = contour_analyzer().analyze(contour, epsillon);
    out<<ca;
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

contour_attribute contour_analyzer::
analyze(std::vector<cv::Point> const &contour,
        double epsillon)
{      
    return analyze(contour, epsillon, buffer_);
}

contour_attribute contour_analyzer::
analyze(std::vector<cv::Point> const &contour,
        double epsillon) const
{
    std::vector<cv::Point> buffer;
    return analyze(contour, epsillon, buffer);
}

contour_attribute contour_analyzer::
analyze(std::vector<cv::Point> const &contour,
        double epsillon,
        std::vector<cv::Point> &buffer) const
{
    contour_attribute ca;

    ca.contour_area_ = cv::contourArea(contour);
    ca.bounding_rect_ = cv::boundingRect(contour);
    ca.bounding_area_ = static_cast<double>(ca.bounding_rect_.area());
    ca.aspect_ratio_ = ca.bounding_rect_.width /
            static_cast<double>(ca.bounding_rect_.height);
    ca.perimeter_ = cv::arcLength(contour, true);
    ca.extent_ = ca.contour_area_/ca.bounding_area_;

    cv::convexHull(contour, buffer);
    ca.solidity_ = ca.contour_area_/cv::contourArea(buffer);

    cv::approxPolyDP(contour, buffer, ca.perimeter_ * epsillon, true);
    ca.poly_size_ = buffer.size();

    return ca;
}

}
