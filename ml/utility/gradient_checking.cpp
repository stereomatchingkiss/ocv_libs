#include "gradient_checking.hpp"

#include <cmath>

namespace ocv{

namespace ml{

gradient_checking::gradient_checking() :
    epsillon_(0.001),
    inaccuracy_(5e-5)
{

}

/**
 * @brief compute the gradient with the gradient\n
 * compute by the gradient descent
 * @param gradient the gradient want to compute, type must be CV_64F
 * @return true if the inaccuracy is low enough, else false
 * @pre (1) rows and channels of the gradient should be 1\n
 * (2) Should call the function <strong class = "paramname">compute_gradient</strong> first\]n
 * (3) type of <strong class = "paramname">gradient</strong> should be CV_64F
 */
bool gradient_checking::compare_gradient(const cv::Mat &gradient) const
{
    CV_Assert(gradient.channels() == 1 && gradient.rows == 1 &&
              gradient.cols == gradient_.cols &&
              gradient.type() == CV_64F);

    //use std::all_of to replace it if this is c++11
    double const *gradient_ptr = gradient_.ptr<double>(0);
    double const *theta_ptr = gradient.ptr<double>(0);
    for(int i  = 0; i != gradient.cols; ++i){
        if(std::abs(gradient_ptr[i] - theta_ptr[i]) >
                inaccuracy_){
            return false;
        }
    }

    return true;
}

void gradient_checking::set_epsillon(double epsillon)
{
    epsillon_ = epsillon;
}

void gradient_checking::set_inaccuracy(double inaccuracy)
{
    inaccuracy_ = inaccuracy;
}

}}
