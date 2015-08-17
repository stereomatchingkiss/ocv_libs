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

    theta.copyTo(theta_buffer_);
    theta_buffer_.convertTo(theta_buffer_, CV_64F);
}

}}
