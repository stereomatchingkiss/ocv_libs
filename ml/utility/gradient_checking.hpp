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
    void initialize(cv::Mat const &theta);

    double epsillon_;

    cv::Mat gradient_;

    cv::Mat theta_buffer_;
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
 *@return the gradients
 */
template<typename UnaryFunc>
cv::Mat const &gradient_checking::
compute_gradient(const cv::Mat &theta,
                 UnaryFunc cost_function)
{
    initialize(theta);
    double *buffer_ptr = theta_buffer_.ptr<double>(0);
    double *gradient_ptr = gradient_.ptr<double>(0);
    double *theta_ptr = theta.ptr<double>(0);
    for(int i = 0; i != theta.cols; ++i){
        *buffer_ptr = *theta_ptr - epsillon_;
        double const Plus = cost_function(theta_buffer_);
        *buffer_ptr = *theta_ptr - epsillon_;
        double const Minus = cost_function(theta_buffer_);
        *gradient_ptr = (Plus - Minus) / (2 * epsillon_);

        *buffer_ptr = *theta_ptr;
        ++buffer_ptr;
        ++gradient_ptr;
        ++theta_ptr;
    }

    return gradient_;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // GRADIENT_CHECKING_H
