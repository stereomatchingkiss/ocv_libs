#include "attribute.hpp"

#include <boost/format.hpp>

#include <iostream>
#include <sstream>

namespace ocv{

void print_contour_attribute(std::vector<cv::Point> const &contour,
                             double epsillon,
                             std::ostream &out)
{
    double const area = cv::contourArea(contour);
    auto const bounding_rect = cv::boundingRect(contour);
    double const aspect_ratio =
            bounding_rect.width / bounding_rect.height;
    double const perimeter = cv::arcLength(contour, true);    
    double const Extent = area/static_cast<double>(bounding_rect.area());

    std::vector<cv::Point> buffer;
    cv::convexHull(contour, buffer);
    double const Solidity = area/cv::contourArea(buffer);

    cv::approxPolyDP(contour, buffer, perimeter * epsillon, true);
    size_t const poly_size = buffer.size();

    std::ostringstream ostr;
    ostr<<boost::format("%=11f|%=11f|%=11f|%=11f|"
                        "%=11f|%=11f")%area%perimeter
          %aspect_ratio%Extent%Solidity%poly_size;
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
