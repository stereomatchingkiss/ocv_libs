#ifndef PERSPECTIVE_TRANSFORM_HPP
#define PERSPECTIVE_TRANSFORM_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * get the mass of the corners
 * @param   begin begin position of the corners
 * @param   end end position of the corners
 * @return  the centers of the corners
 */
template<typename InputIter>
inline
auto corners_center(InputIter begin, InputIter end)->decltype(*begin)
{
    return std::accumulate(begin, end, decltype(*begin)())
            * (1. / (end - begin));
}

/**
 * get the mass of the corners
 * @param   input position of the corsers
 * @return  the centers of the corners
 */
template<typename T>
inline
auto corners_center(T const &input)->decltype(*std::begin(input))
{
    return corners_center(std::begin(input), std::end(input));
}

/**
 * sort the corners of the rectangle to top left, top right,
 * bottom right, bottom left.
 * users have to make sure the points of the corners are valid(is a rect),
 * else the results are undefined
 * @param begin begin position of corners
 * @param end end position of corners
 * @param out Output iterator to the initial position
 * in the destination sequence.
 * @warning end - begin must equal to 4, the sequence point by
 * out must have enough space to store 4 points
 */
template<typename InputIter, typename OutputIter>
void sort_corners(InputIter begin, InputIter end, OutputIter out)
{
    std::array<decltype(*begin),2> top, bot;
    auto const center = corners_center(begin, end);
    auto top_iter = top.begin();
    auto bot_iter = bot.begin();
    while(begin != end){
        if(begin->y < center.y){
            *top_iter = corners[i];
            ++top_iter;
        }else{
            *bot_iter = corners[i];
            ++bot_iter;
        }
    }

    *out = top[0].x > top[1].x ? top[1] : top[0]; //top left
    ++out;
    *out = top[0].x > top[1].x ? top[0] : top[1]; //top right
    ++out;
    *out = bot[0].x > bot[1].x ? bot[0] : bot[1]; //bottom right
    ++out;
    *out = bot[0].x > bot[1].x ? bot[1] : bot[0]; //bottom left
    ++out;
}

/**
 * sort the corners of the rectangle to top left, top right,
 * bottom right, bottom left.
 * users have to make sure the points of the corners are valid(is a rect),
 * else the results are undefined
 * @param corners Sequence of corners, the size must be 4
 * @param out Output iterator to the initial position
 * in the destination sequence.
 */
template<typename T, typename OutIter>
inline
void sort_corners(T const &corners, OutIter out)
{
    sort_corners(std::begin(corners), std::end(corners), out);
}

/**
 * find euclidean distance between two points
 * @param lhs first point
 * @param rhs second point
 * @return euclidean distance of two points
 */
template<typename U = double, typename T, typename V>
inline
U point_euclidean_dist(T const &lhs, V const &rhs)
{
    U const width_diff = static_cast<U>(lhs.x - rhs.x);
    U const height_diff = static_cast<U>(lhs.y - rhs.y);

    return std::sqrt(width_diff * width_diff +
                     height_diff * height_diff);
}

/**
 *transform the input to bird eyes view
 *@param input input image want to transform to bird eyes view
 *@param corners corners(4 points) of the input image
 *@param output bird eyes view of input
 */
template<typename T>
void four_points_transform(cv::Mat const &input,
                           T const &input_corners,
                           cv::Mat &output)
{
    using value_type = decltype(input_corners[0]);

    value_type sorted_input[4];
    sort_corners(input_corners, std::begin(sorted_input));

    size_t const max_width =
            std::max(point_euclidean_dist<float>(sorted_input[0], sorted_input[1]),
                     point_euclidean_dist<float>(sorted_input[2], sorted_input[3]));
    size_t const max_height =
            std::max(point_euclidean_dist<float>(sorted_input[0], sorted_input[1]),
                     point_euclidean_dist<float>(sorted_input[2], sorted_input[3]));

    cv::Point2f const trans_corners[]
    {
        {0, 0},
        {max_width - 1, 0},
        {max_width - 1, max_height - 1},
        {0, max_height - 1}
    };

    auto const trans_mat =
            cv::getPerspectiveTransform(sorted_input, trans_corners);
    cv::warpPerspective(input, output, trans_mat, output.size());
}

/**
 *transform the input to bird eyes view
 *@param input input image want to transform to bird eyes view
 *@param corners corners(4 points) of the input image
 *@return output bird eyes view of input
 */
template<typename T>
inline
cv::Mat four_points_transform(cv::Mat const &input,
                              T const &input_corners)
{
  cv::Mat output;
  four_points_transform(input, input_corners, output);

  return output;
}


}; /*! @} End of Doxygen Groups*/

#endif // PERSPECTIVE_TRANSFORM_HPP


