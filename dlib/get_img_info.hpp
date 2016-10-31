#ifndef OCV_DLIB_GET_IMG_INFO_HPP
#define OCV_DLIB_GET_IMG_INFO_HPP

#include <dlib/geometry.h>

#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup odlib
 *  @{
 */
namespace odlib{

void get_imglab_xml_info(std::vector<std::string> &img_name,
                         std::vector<std::vector<dlib::rectangle>> &location,
                         std::string const &file_name);


} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_DLIB_GET_IMG_NAME_AND_POSITION_HPP
