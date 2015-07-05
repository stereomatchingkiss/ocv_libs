#ifndef IO_IMG_HPP
#define IO_IMG_HPP

#include <QImage>
#include <QString>

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

cv::Mat read_cv_mat(QString const &file);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // IO_IMG_HPP

