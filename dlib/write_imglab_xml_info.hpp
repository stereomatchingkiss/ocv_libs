#ifndef WRITE_DLIB_IMGLAB_XML_INFO_HPP
#define WRITE_DLIB_IMGLAB_XML_INFO_HPP

#include <dlib/geometry.h>
#include <dlib/image_processing.h>

#include <string>
#include <type_traits>
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

/**
 *@overload overload of write_imglab_xml, this function zip img_name and roi together
 */
template<typename T>
void write_imglab_xml(std::vector<T> const &input,
                      std::string const &prepend_folder,
                      std::string const &output_name)
{

    using type_0 = std::tuple<std::string, std::vector<dlib::rectangle>>;
    using type_1 = std::tuple<std::string, std::vector<dlib::mmod_rect>>;
    using type_2 = std::pair<std::string, std::vector<dlib::rectangle>>;
    using type_3 = std::pair<std::string, std::vector<dlib::mmod_rect>>;
    static_assert(std::is_same<T, type_0>::value || std::is_same<T, type_1>::value ||
                  std::is_same<T, type_2>::value || std::is_same<T, type_3>::value,
                  "T == std::tuple<std::string, std::vector<dlib::rectangle>> || "
                  "T == std::tuple<std::string, std::vector<dlib::mmod_rect>> || "
                  "T == std::pair<std::string, std::vector<dlib::rectangle>> || "
                  "T == std::pair<std::string, std::vector<dlib::mmod_rect>>");

    using roi_vtype = typename std::decay<decltype(std::get<1>(input[0]))>::type;
    std::vector<std::string> img_name;
    std::vector<roi_vtype> roi;

    for(size_t i = 0; i != input.size(); ++i){
        img_name.emplace_back(std::get<0>(input[i]));
        roi.emplace_back(std::get<1>(input[i]));
    }

    write_imglab_xml(img_name, roi, prepend_folder, output_name);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // WRITE_DLIB_IMGLAB_XML_INFO_HPP
