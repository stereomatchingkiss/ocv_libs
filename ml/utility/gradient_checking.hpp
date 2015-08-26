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
 * to check the results of gradient descent like algorithm.
 */

class gradient_checking
{
public:
    gradient_checking();

    bool compare_gradient(cv::Mat const &gradient) const;

    template<typename T, typename UnaryFunc>
    cv::Mat const& compute_gradient(cv::Mat const &theta,
                                    UnaryFunc func);

    void set_epsillon(double epsillon);
    void set_inaccuracy(double inaccuracy);

private:
    double epsillon_;

    cv::Mat gradient_;

    double inaccuracy_;

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
 * one channel
 *@return the gradients
 */
template<typename T, typename UnaryFunc>
cv::Mat const &gradient_checking::
compute_gradient(const cv::Mat &theta,
                 UnaryFunc cost_function)
{    
    theta.copyTo(theta_buffer_);
    gradient_.create(theta.rows,
                     theta.cols,
                     theta.type());
    for(int row = 0; row != theta.rows; ++row){
        for(int col = 0; col != theta.cols; ++col){
            auto const OriValue = theta_buffer_.at<T>(row, col);
            theta_buffer_.at<T>(row, col) =
                    OriValue + epsillon_;
            auto const Plus = cost_function(theta_buffer_);
            theta_buffer_.at<T>(row, col) =
                    OriValue - epsillon_;
            auto const Minus = cost_function(theta_buffer_);
            gradient_.at<T>(row, col) =
                    (Plus - Minus) / (2 * epsillon_);

            theta_buffer_.at<T>(row, col) = OriValue;
        }
    }

    return gradient_;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // GRADIENT_CHECKING_H
