#ifndef CV_TO_EIGEN_HPP
#define CV_TO_EIGEN_HPP

#include <Eigen/Dense>

/*! \file cv_to_eigen.hpp
    \brief collect some tools to transform cv::Mat to\n
    Eigen matrix
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup eigen
 *  @{
 */
namespace eigen{

template<typename T = double>
using CV2Eigen =
Eigen::Map<Eigen::Matrix<T,
Eigen::Dynamic,
Eigen::Dynamic,Eigen::RowMajor> >;

using CV2EigenD = CV2Eigen<double>;

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // CV_TO_EIGEN_HPP

