#ifndef OCV_DLIB_CV_TO_DLIB_HPP
#define OCV_DLIB_CV_TO_DLIB_HPP

#include <opencv2/core.hpp>

#include <dlib/image_processing.h>

#include <algorithm>
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

template<typename T = int>
inline
cv::Rect_<T> rect_to_cv_rect(dlib::rectangle const &input)
{
    return cv::Rect_<T>(input.left(), input.top(),
                        input.width(), input.height());
}

template<typename InPutIter1, typename OutputIter2>
inline
void rect_to_cv_rect(InPutIter1 beg1, InPutIter1 end1,
                     OutputIter2 beg2)
{
    std::transform(beg1, end1, beg2,
                   [](auto const &val)
    {
        using value_type = typename std::iterator_traits<OutputIter2>::value_type;
        return rect_to_cv_rect<value_type::value_type>(val);
    });
}

template<typename T>
inline
void rect_to_cv_rect(std::vector<dlib::rectangle> const &input,
                     std::vector<cv::Rect_<T>> &output)
{
    output.resize(input.size());
    rect_to_cv_rect(std::begin(input), std::end(input),
                    std::begin(output));
}

template<typename T = int, typename InputIter>
inline
std::vector<cv::Rect_<T>> rect_to_cv_rect(InputIter beg, InputIter end)
{
    std::vector<cv::Rect_<T>> results;
    results.resize(std::distance(beg, end));

    return rect_to_cv_rect(beg, end, std::begin(results));
}

template<typename T = int>
inline
std::vector<cv::Rect_<T>> rect_to_cv_rect(std::vector<dlib::rectangle> const &input)
{
    std::vector<cv::Rect_<T>> results;
    rect_to_cv_rect<T>(input, results);

    return results;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_DLIB_CV_TO_DLIB_HPP
