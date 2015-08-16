#ifndef GRADIENT_CHECKING_H
#define GRADIENT_CHECKING_H

#include <opencv2/core.hpp>

#include <functional>

/*! \file gradient_checking.hpp
    \brief check the results of gradient descent like algorithm
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup ml
 *  @{
 */
namespace ml{

/**
 * @brief The gradient_checking class which use\n
 * to check the results of gradient descent like algorithm.\n
 * This class intent to work with c++98
 */

class gradient_checking
{
public:
    gradient_checking();

    template<typename UnaryFunc>
    cv::Mat const& compute_gradient(cv::Mat const &theta,
                                    UnaryFunc func);

    void set_epsillon(double epsillon);

private:
    double epsillon_;

    cv::Mat gradient_;

    cv::Mat theta_minus_;
    cv::Mat theta_plus_;
};

/**
 *@brief compute the gradient by gradient checking.
 *@param theta The "weights" which need to optimize by\n
 * the gradient descent like algo
 *@param cost_function The cost function of the algo, this\n
 * function only accept one parameter--cv::Mat which contains\n
 * the (theta[i] + epsillon) or (theta[i] - epsillon)
 *@pre <strong class = "paramname">theta</strong> must be\n
 * one channel and one row
 */
template<typename UnaryFunc>
cv::Mat const &gradient_checking::
compute_gradient(const cv::Mat &theta,
                 UnaryFunc cost_function)
{
    CV_Assert(theta.channels() == 1 &&
              theta.rows == 1);

    theta.copyTo(theta_minus_);
    theta.copyTo(theta_plus_);
    theta_minus_.convertTo(theta_minus_, CV_64F);
    theta_plus_.convertTo(theta_plus_, CV_64F);
    double *gradient_ptr = gradient_.ptr<double>(0);
    double *minus_ptr = theta_minus_.ptr<double>(0);
    double *plus_ptr = theta_plus_.ptr<double>(0);
    double *theta_ptr = theta.ptr<double>(0);
    for(int i = 0; i != theta.cols; ++i){
        *minus_ptr = *theta_ptr - epsillon_;
        *plus_ptr = *theta_ptr + epsillon_;
        *gradient_ptr = (cost_function(theta_plus_) - cost_function(theta_minus_)) /
                2 * epsillon_;
        *minus_ptr = *theta_ptr;
        *plus_ptr = *theta_ptr;

        ++gradient_ptr;
        ++minus_ptr;
        ++plus_ptr;
        ++theta_ptr;
    }

    return gradient_;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // GRADIENT_CHECKING_H
