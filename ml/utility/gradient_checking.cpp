#include "gradient_checking.hpp"

namespace ocv{

namespace ml{

gradient_checking::gradient_checking() :
    epsillon_(0.001)
{

}

void gradient_checking::set_epsillon(double epsillon)
{
    epsillon_ = epsillon;
}

void gradient_checking::initialize(const cv::Mat &theta)
{
    CV_Assert(theta.channels() == 1 &&
              theta.rows == 1);

    theta.copyTo(theta_minus_);
    theta.copyTo(theta_plus_);
    theta_minus_.convertTo(theta_minus_, CV_64F);
    theta_plus_.convertTo(theta_plus_, CV_64F);
}

}}
