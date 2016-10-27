#ifndef OCV_DLIB_CV_TO_DLIB_HPP
#define OCV_DLIB_CV_TO_DLIB_HPP

#include <opencv2/core.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>

#include <algorithm>
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

template<typename T = int, typename DlibRect>
inline
cv::Rect_<T> rect_to_cv_rect(DlibRect const &input)
{    
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    return cv::Rect_<T>(input.left(), input.top(),
                        input.width(), input.height());
}

template<typename InPutIter1, typename OutputIter2>
inline
void rect_to_cv_rect(InPutIter1 beg1, InPutIter1 end1,
                     OutputIter2 beg2)
{
    using value_type = typename std::iterator_traits<OutputIter2>::value_type;
    static_assert(std::is_arithmetic<value_type>::value, "value_type of "
                                                         "OutputIter2 must be arithmetic");

    std::transform(beg1, end1, beg2,
                   [](auto const &val)
    {
        return rect_to_cv_rect<value_type::value_type>(val);
    });
}

template<typename T, typename DlibRect>
inline
void rect_to_cv_rect(std::vector<DlibRect> const &input,
                     std::vector<cv::Rect_<T>> &output)
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    output.resize(input.size());
    rect_to_cv_rect(std::begin(input), std::end(input),
                    std::begin(output));
}

template<typename T = int, typename InputIter>
inline
std::vector<cv::Rect_<T>> rect_to_cv_rect(InputIter beg, InputIter end)
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    std::vector<cv::Rect_<T>> results;
    results.resize(std::distance(beg, end));

    return rect_to_cv_rect(beg, end, std::begin(results));
}

template<typename T = int, typename DlibRect>
inline
std::vector<cv::Rect_<T>> rect_to_cv_rect(std::vector<DlibRect> const &input)
{
    std::vector<cv::Rect_<T>> results;
    rect_to_cv_rect<T>(input, results);

    return results;
}

template<typename ImgType>
ImgType copy_cvmat_to_dlib_mat(cv::Mat &input, bool release_after_copy = true)
{
    static_assert(std::is_class<ImgType>::value, "ImgType should be a class");

    using pixel_type = typename dlib::image_traits<ImgType>::pixel_type;
    cv_image<pixel_type> cimg(input);
    ImgType output;
    dlib::assign_image(output, cimg);
    if(release_after_copy){
        input.release();
    }

    return output;
}

template<typename ImgType>
void copy_cvmat_to_dlib_mat(cv::Mat &input, ImgType &output, bool release_after_copy = true)
{
    static_assert(std::is_class<ImgType>::value, "ImgType should be a class");
    output = copy_cvmat_to_dlib_mat<ImgType>(input, release_after_copy);
}

template<typename ImgType, typename CVMats>
std::vector<ImgType> copy_cvmat_to_dlib_mat(CVMats &cv_mats, bool release_after_copy = true)
{    
    static_assert(std::is_class<CVMats>::value, "CVMats should be a class");

    std::vector<ImgType> results;
    for(auto &cv_mat : cv_mats){
        results.emplace_back(copy_cvmat_to_dlib_mat<ImgType>(cv_mat, release_after_copy));
    }

    return results;
}

template<typename ImgsType, typename CVMats>
void copy_cvmat_to_dlib_mat(CVMats &cv_mats, ImgsType &output, bool release_after_copy = true)
{
    static_assert(std::is_class<ImgsType>::value, "ImgsType should be a class");
    static_assert(std::is_class<CVMats>::value, "CVMats should be a class");

    using img_type = std::decay<decltype(output[0])>::type;
    std::vector<img_type> results = copy_cvmat_to_dlib_mat<img_type>(cv_mats, release_after_copy);
    std::move(std::begin(results), std::end(results),
              std::back_inserter(output));
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_DLIB_CV_TO_DLIB_HPP
