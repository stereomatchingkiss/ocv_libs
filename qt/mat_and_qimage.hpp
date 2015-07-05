#ifndef OPENCVTOQT_HPP
#define OPENCVTOQT_HPP

/*! \file mat_and_qimage.h
    \brief conversion between qimage and mat
*/

#include <QImage>

#include <opencv2/core.hpp>

/*!
 *  \addtogroup ocv
 *  @{
 *  //ocv mean "opencv"
 */
namespace ocv{

/*!
 *  \addtogroup qt
 *  @{
 *  //handle the chores related with opencv and qt
 */
namespace qt{

QImage mat_to_qimage_cpy(cv::Mat const &mat, bool swap = true);

QImage mat_to_qimage_ref(cv::Mat &mat, bool swap = true);

cv::Mat qimage_to_mat_cpy(QImage const &img, bool swap = true);

cv::Mat qimage_to_mat_ref(QImage &img, bool swap = true);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OPENCVTOQT_HPP
