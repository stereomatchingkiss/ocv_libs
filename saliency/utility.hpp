#ifndef OCV_SALIENCY_UTILITY_HPP
#define OCV_SALIENCY_UTILITY_HPP

#include <opencv2/core.hpp>

#include <cmath>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup saliency
 *  @{
 */
namespace saliency{

namespace details{

template<typename T>
inline
void find_intersect_range(T pt1, T pt2,
                          T pt3, T pt4,
                          T &intersection_start,
                          T &intersection_len)
{
    intersection_start = pt1 < pt2 ?
                pt2 : pt1;
    intersection_len = pt3 < pt4 ?
                intersection_start - pt3 :
                intersection_start - pt4;
}

/**
 *overload of calculate_iou, convert box1 and box2
 *to appropriate type once and for all
 */
template<typename T>
T calculate_iou_impl(cv::Vec<T, 4> const &box1,
                     cv::Vec<T, 4> const &box2)
{
    T const width1 = std::abs(box1[3] - box1[1]);
    T const width2 = std::abs(box2[3] - box2[1]);
    T const height1 = std::abs(box1[2] - box1[0]);
    T const height2 = std::abs(box2[2] - box2[0]);

    T const center1_x = width1 / T(2.0) + box1[1];
    T const center1_y = height1 / T(2.0) + box1[0];
    T const center2_x = width2 / T(2.0) + box2[1];
    T const center2_y = height2 / T(2.0) + box2[0];
    T const diff_center_x = std::abs(center1_x - center2_x);
    T const diff_center_y = std::abs(center1_y - center2_y);

    T intersection_start_x = 0;
    T intersection_width = 0;
    T intersection_start_y = 0;
    T intersection_height = 0;

    if (diff_center_x - (width1 / 2 + width2 / 2) < 0){ //they intersect
        details::find_intersect_range(box1[1], box2[1],
                box1[3], box2[3],
                intersection_start_x, intersection_width);
    }else{ //they don't intersect
        return 0;
    }

    if (diff_center_y - (height1 / 2 + height2 / 2) < 0){ //they intersect
        details::find_intersect_range(box1[0], box2[0],
                box1[2], box2[2],
                intersection_start_y, intersection_height);
    }else{ //they don't intersect
        return 0;
    }

    T const intersection_area =
            intersection_width*intersection_height;
    T const union_area =
            width1*height1 + width2*height2 - intersection_area;

    if (union_area == 0){
        return 0;
    }

    return intersection_area / union_area;
}


}

/**
 *find intersection over Union(IoU)
 *@param box1 save point1 and point2 of box1,
 * points store as (y1,x1,y2,x2), y refer to row
 *@param box2 save point1 and point2 of box2,
 * points store as (y1,x1,y2,x2), y refer to row
 */
template<typename T,typename U = float>
inline
U calculate_iou(cv::Vec<T, 4> const &box1,
                cv::Vec<T, 4> const &box2)
{
    static_assert(std::is_floating_point<U>::value,
                  "U should be floating point");

    return details::calculate_iou_impl(cv::Vec<U,4>(box1),
                                       cv::Vec<U,4>(box2));
}

template<typename T,typename U = float>
inline
U calculate_iou(cv::Rect_<T> const &box1,
                cv::Rect_<T> const &box2)
{
    static_assert(std::is_floating_point<U>::value,
                  "U should be floating point");

    cv::Rect_<U> const b1 = box1;
    cv::Rect_<U> const b2 = box2;

    return calculate_iou(cv::Vec<U,4>(b1.y, b1.x,
                                      b1.y + b1.height,
                                      b1.x + b1.width),
                         cv::Vec<U,4>(b2.y, b2.x,
                                      b2.y + b2.height,
                                      b2.x + b2.width));
}


} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_SALIENCY_UTILITY_HPP
