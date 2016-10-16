#ifndef OCV_CORE_RECT_UTILITY_HPP
#define OCV_CORE_RECT_UTILITY_HPP

#include <opencv2/core.hpp>

#include <cmath>
#include <type_traits>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

template<typename T, typename U = typename std::common_type<T,float>::type>
inline
U diagonal_euclidean_dist(cv::Rect_<T> const &p);

template<typename T, typename U = typename std::common_type<T,float>::type>
inline
U center_euclidean_dist(cv::Rect_<T> const &p, cv::Rect_<T> const &q);

template<typename T>
inline
cv::Point_<T> rect_center(cv::Rect_<T> const &rect);

template<typename T, typename U>
inline
U center_euclidean_dist(cv::Rect_<T> const &p, cv::Rect_<T> const &q)
{
    static_assert(std::is_floating_point<U>::value, "U must be floating point");

    auto const diff = rect_center(p) - rect_center(q);
    return std::sqrt(diff.x*diff.x + diff.y*diff.y);
}

template<typename T, typename U>
inline
U diagonal_euclidean_dist(cv::Rect_<T> const &p)
{
    return static_cast<U>(std::sqrt(p.width * p.width + p.height * p.height));
}

template<typename T>
inline
cv::Point_<T> rect_center(cv::Rect_<T> const &rect)
{
    return {rect.x + rect.width / 2, rect.y + rect.height / 2};
}


} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_RECT_UTILITY_HPP
