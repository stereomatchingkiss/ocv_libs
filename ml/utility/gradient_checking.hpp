#ifndef GRADIENT_CHECKING_H
#define GRADIENT_CHECKING_H

#include "../../core/utility.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <functional>
#include <type_traits>

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
    template<typename T>
    bool compare_gradient(cv::Mat_<T> const &grad_1,
                          cv::Mat_<T> const &grad_2,
                          double epsillon = 1e-5) const;

    template<typename EigenMat>
    bool compare_gradient(EigenMat const &grad_1,
                          EigenMat const &grad_2,
                          double epsillon = 1e-5) const;

    template<typename T, typename UnaryFunc>
    cv::Mat compute_gradient(cv::Mat_<T> const &theta,
                             UnaryFunc cost_function,
                             double epsillon = 1e-2) const;

    template<typename EigenMat, typename UnaryFunc>
    EigenMat compute_gradient(EigenMat const &theta,
                              UnaryFunc cost_function,
                              double epsillon = 1e-2) const;
};

template<typename T>
bool gradient_checking::
compare_gradient(cv::Mat_<T> const &grad_1,
                 cv::Mat_<T> const &grad_2,
                 double epsillon) const
{
    static_assert(std::is_arithmetic<T>::value,
                  "T must be arithmetic type");

    return compare_channels<T>(grad_1, grad_2,
                               [&](T lhs, T rhs)
    {
        return std::abs(lhs - rhs) < epsillon;
    });
}

/**
 * @brief compute two gradient, make sure their different are small
 * @param grad_1 the gradient want to compute with grad_2
 * @param grad_2 the gradient want to compute with grad_1
 * @param epsillon The inaccuracy between two epsillon
 * @return true if the inaccuracy is low enough, else false
 */
template<typename EigenMat>
bool gradient_checking::
compare_gradient(EigenMat const &grad_1,
                 EigenMat const &grad_2,
                 double epsillon) const
{
    static_assert(!std::is_same<cv::Mat, EigenMat>::value,
                  "This function do not support opencv yet");
    EigenMat const Diff =
            (grad_1.array() - grad_2.array()).abs();
    return (Diff.array() < epsillon ).all();
}

/**
 *@brief compute the gradient by gradient checking.
 *@param theta The "weights" which need to optimize by\n
 * the gradient descent like algo
 *@param cost_function The cost function of the algo, this\n
 * function only accept one parameter--cv::Mat which contains\n
 * the (theta[i] + epsillon) or (theta[i] - epsillon)
 @param epsillon The inaccuracy between two epsillon
 *@pre <strong class = "paramname">theta</strong> must be\n
 * one channel
 *@return the gradients
 */
template<typename T, typename UnaryFunc>
cv::Mat gradient_checking::
compute_gradient(const cv::Mat_<T> &theta,
                 UnaryFunc cost_function,
                 double epsillon) const
{    
    static_assert(std::is_arithmetic<T>::value,
                  "T must be arithmetic type");

    CV_Assert(theta.channels() == 1);

    cv::Mat_<T> theta_buffer;
    theta.copyTo(theta_buffer);
    cv::Mat gradient;
    gradient.create(theta.rows,
                    theta.cols,
                    theta.type());
    for(int row = 0; row != theta.rows; ++row){
        for(int col = 0; col != theta.cols; ++col){
            auto const OriValue = theta_buffer.at<T>(row, col);
            theta_buffer.at<T>(row, col) =
                    OriValue + epsillon;
            auto const Plus = cost_function(theta_buffer);
            theta_buffer.at<T>(row, col) =
                    OriValue - epsillon;
            auto const Minus = cost_function(theta_buffer);
            gradient.at<T>(row, col) =
                    (Plus - Minus) / (2 * epsillon);

            theta_buffer.at<T>(row, col) = OriValue;
        }
    }

    return gradient;
}

/**
 *@brief compute the gradient by gradient checking.
 *@param theta The "weights" which need to optimize by\n
 * the gradient descent like algo
 *@param cost_function The cost function of the algo, this\n
 * function only accept one parameter--cv::Mat which contains\n
 * the (theta[i] + epsillon) or (theta[i] - epsillon)
 *@param epsillon The inaccuracy between two epsillon
 *@pre <strong class = "paramname">theta</strong> must be\n
 * one channel
 *@return the gradients
 */
template<typename EigenMat, typename UnaryFunc>
EigenMat gradient_checking::
compute_gradient(EigenMat const &theta,
                 UnaryFunc cost_function,
                 double epsillon) const
{
    EigenMat theta_buffer = theta;
    EigenMat gradient = EigenMat::Zero(theta_buffer.rows(),
                                       theta_buffer.cols());
    for(int row = 0; row != theta.rows(); ++row){
        for(int col = 0; col != theta.cols(); ++col){
            auto const OriValue = theta_buffer(row, col);
            theta_buffer(row, col) =
                    OriValue + epsillon;
            auto const Plus = cost_function(theta_buffer);
            theta_buffer(row, col) =
                    OriValue - epsillon;
            auto const Minus = cost_function(theta_buffer);
            gradient(row, col) =
                    (Plus - Minus) / (2 * epsillon);

            theta_buffer(row, col) = OriValue;
        }
    }

    return gradient;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // GRADIENT_CHECKING_H
