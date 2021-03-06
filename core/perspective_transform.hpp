#ifndef PERSPECTIVE_TRANSFORM_HPP
#define PERSPECTIVE_TRANSFORM_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <iostream>
#include <iterator>
#include <numeric>
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
typename std::iterator_traits<InputIter>::value_type
corners_center(InputIter begin, InputIter end)
{
    using value_type = std::iterator_traits<InputIter>::value_type;
    return std::accumulate(begin, end, value_type())
            * (1. / std::distance(begin, end));
}

/**
 * get the mass of the corners
 * @param   input position of the corsers
 * @return  the centers of the corners
 */
template<typename T>
inline
auto corners_center(T const &input)
{
    return corners_center(std::begin(input), std::end(input));
}

template<typename T, size_t N>
inline
auto corners_center(T const (&input)[N])
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
    using value_type = std::iterator_traits<InputIter>::value_type;

    std::array<value_type, 4> buffer;
    std::copy(begin, end, std::begin(buffer));
    std::sort(std::begin(buffer), std::end(buffer),
              [](auto const &lhs, auto const &rhs)
    {
       return lhs.x < rhs.x;
    });    

    //find out top left and bottom left
    value_type bl;
    if(buffer[0].y < buffer[1].y){
        *out = buffer[0];
        bl = buffer[1];
    }else{
        *out = buffer[1];
        bl = buffer[0];
    }
    ++out;

    //find out top right and bottom right
    if(buffer[2].y < buffer[3].y){
      *out = buffer[2]; ++out;
      *out = buffer[3];
    }else{
        *out = buffer[3]; ++out;
        *out = buffer[2];
    }
    ++out;
    *out = bl;
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

template<typename T, typename OutIter, size_t N>
inline
void sort_corners(T const (&corners)[N], OutIter out)
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
 *@param output bird eyes view of input
 *@param input_corners corners of the input image, sorted by
 * top left, top right, bottom right, bottom left
 *@param output_corners corners of the output image, sorted by
 * top left, top right, bottom right, bottom left
 */
template<typename T, typename U>
void four_points_transform(cv::Mat const &input,
                           cv::Mat &output,
                           T const (&input_corners)[4],
                           U const (&output_corners)[4])
{    
    auto const trans_mat =
            cv::getPerspectiveTransform(input_corners, output_corners);
    cv::warpPerspective(input, output, trans_mat,
    {static_cast<int>(output_corners[2].x+1),
     static_cast<int>(output_corners[2].y+1)});
}

/**
 *transform the input to bird eyes view
 *@param input input image want to transform to bird eyes view
 *@param output bird eyes view of input
 *@param corners corners(4 points) of the input image
 */
template<typename T>
void four_points_transform(cv::Mat const &input,
                           cv::Mat &output,
                           T const (&input_corners)[4])
{        
    T sorted_input[4];
    sort_corners(input_corners, std::begin(sorted_input));
    //std::copy(std::begin(input_corners), std::end(input_corners),
    //          std::begin(sorted_input));
    /**
     * The geometry of the coordinates
     * sorted_input[0]      sorted_input[1]
     * sorted_input[3]      sorted_input[2]
     */

    float const max_width =
            std::max(point_euclidean_dist<float>(sorted_input[0], sorted_input[1]),
                     point_euclidean_dist<float>(sorted_input[2], sorted_input[3]));
    float const max_height =
            std::max(point_euclidean_dist<float>(sorted_input[0], sorted_input[3]),
                     point_euclidean_dist<float>(sorted_input[1], sorted_input[2]));

    T const trans_corners[]
    {
        {0, 0},
        {max_width - 1, 0},
        {max_width - 1, max_height - 1},
        {0, max_height - 1}
    };

    four_points_transform(input, output, sorted_input, trans_corners);
}

/**
 *transform the input to bird eyes view
 *@param input input image want to transform to bird eyes view
 *@param output bird eyes view of input
 *@param corners corners(4 points) of the input image
 */
template<typename T>
inline
void four_points_transform(cv::Mat const &input,
                           cv::Mat &output,
                           T const &input_corners)
{
    using value_type = decltype(input_corners[0]);

    value_type copy_input[4];
    std::copy(std::begin(input_corners), std::end(input_corners),
              std::begin(copy_input));
    four_points_transform(input, output, copy_input);
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
                              T const (&input_corners)[4])
{
  cv::Mat output;
  four_points_transform(input, output, input_corners);

  return output;
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
  four_points_transform(input, output, input_corners);

  return output;
}


}; /*! @} End of Doxygen Groups*/

#endif // PERSPECTIVE_TRANSFORM_HPP


