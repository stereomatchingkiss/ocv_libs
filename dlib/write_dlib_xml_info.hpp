#ifndef WRITE_DLIB_XML_INFO_HPP
#define WRITE_DLIB_XML_INFO_HPP

#include <dlib/geometry.h>
#include <dlib/image_processing.h>

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

/**
 * @brief write the image name and bounding box to the xml format can be
 * recognize by imglab
 * @param image_location location of the images
 * @param rects bounding boxes
 * @param prepend_folder <image file="[prepend_folder]/mov_001_007585.jpeg">
 * @param output_name name of the output xml
 */
void write_imglab_xml(std::vector<std::string> const &image_location,
                    std::vector<std::vector<dlib::rectangle>> const &rects,
                    std::string const &prepend_folder,
                    std::string const &output_name);

/**
 * @brief overload of write_dlib_xml, everything are same except this one
 * accept mmod_rect but not rect
 */
void write_imglab_xml(std::vector<std::string> const &image_location,
                    std::vector<std::vector<dlib::mmod_rect>> const &rects,
                    std::string const &prepend_folder,
                    std::string const &output_name);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // WRITE_DLIB_XML_INFO_HPP
