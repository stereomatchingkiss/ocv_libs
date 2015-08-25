#ifndef CV_TO_EIGEN_HPP
#define CV_TO_EIGEN_HPP

#include <opencv2/core.hpp>

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

template<typename T>
using MatRowMajor =
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
Eigen::RowMajor>;

template<typename T>
void cv2eigen_cpy(cv::Mat const &input, MatRowMajor<T> &output){
    output.resize(input.rows, input.cols);
    cv::Mat dst(input.rows, input.cols, cv::DataType<T>::type,
                output.data(),
                (size_t)(output.stride()*sizeof(double)));
    input.convertTo(dst, dst.type());
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // CV_TO_EIGEN_HPP

