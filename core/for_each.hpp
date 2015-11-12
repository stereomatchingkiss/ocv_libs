#ifndef OCV_GENERICFOREACH_HPP
#define OCV_GENERICFOREACH_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv
{

template<typename T, typename UnaryFunc, typename Mat>
UnaryFunc for_each_channels(Mat &&input, UnaryFunc func);

/**
 *@brief process each block of the image
 *@param input the input image
 *@param block the size of the subimage
 *@param func functor accept three parameters.\n
 * col : x-axis position of the image(not block)\n
 * row : y-axis position of the image(not block)\n
 * subimage : the subimage of image
 *@param stride the step when sliding the subimage
 */
template<typename TerFunc>
TerFunc
for_each_block(cv::Mat const &input,
               cv::Size2i const &block,
               TerFunc func,
               cv::Size2i const &stride)
{
    for(int row = 0; row <= input.rows - block.height;
        row += stride.height){
        for(int col = 0; col <= input.cols - block.width;
            col += stride.width){
            func(row, col, input(cv::Rect{col, row,
                                          block.width,
                                          block.height}));
        }
    }

    return func;
}

template<typename TerFunc>
inline TerFunc
for_each_block(cv::Mat const &input,
               cv::Size2i const &block,
               TerFunc func)
{
    return for_each_block(input, block, func, block);
}

/**
 *@brief process each block of the images
 *@param input_1 first input image
 *@param input_2 second input image
 *@param block the size of the subimage
 *@param func functor accept four parameters.\n
 * col : x-axis position of the image(not block)\n
 * row : y-axis position of the image(not block)\n
 * subimage_1 : the subimage of first input image
 * subimage_2 : the subimage of second input image
 *@param stride the step when sliding the subimage
 */
template<typename QuatFunc>
QuatFunc
for_each_block(cv::Mat const &input_1,
               cv::Mat const &input_2,
               cv::Size2i const &block,
               QuatFunc func,
               cv::Size2i const &stride = {1, 1})
{
    CV_Assert(input_1.rows / stride.height ==
              input_2.rows / stride.height);
    CV_Assert(input_1.cols / stride.width ==
              input_2.cols / stride.width);

    for(int row = 0; row <= input_1.rows - block.height;
        row += stride.height){
        for(int col = 0; col <= input_1.cols - block.width;
            col += stride.width){
            func(row, col,
                 input_1(cv::Rect{col, row,
                                  block.width,
                                  block.height}),
                 input_2(cv::Rect{col, row,
                                  block.width,
                                  block.height}));
        }
    }

    return func;
}

/**
 *@brief apply stl like for_each algorithm on a channel
 *
 * @param T : the type of the channel(ex, uchar, float, double and so on)
 * @param channel : the channel need to apply for_each algorithm
 * @param func : Unary function that accepts an element in the range as argument
 *
 *@return :
 *  return func
 */
template<typename T, typename UnaryFunc, typename Mat>
inline 
UnaryFunc for_each_channel(Mat &&input, int channel, UnaryFunc func)
{    
    if(input.channels() == 1){
        return for_each_channels<T>(std::forward<Mat>(input), func);
    }

    int const size = input.total();
    int const channels = input.channels();
    auto input_ptr = input.template ptr<T>(0) + channel;
    for(int i = 0; i != size; ++i){
        func(*input_ptr);
        input_ptr += channels;
    }

    return func;
}

/**
 *@brief apply stl like for_each algorithm on a channel
 *
 * @param T : the type of the channel(ex, uchar, float, double and so on)
 * @param func : Unary function that accepts an element in the range as argument
 *
 *@return :
 *  return func
 */
template<typename T, typename UnaryFunc, typename Mat>
UnaryFunc for_each_channels(Mat &&input, UnaryFunc func)
{
    int rows = input.rows;
    int cols = input.cols;

    if(input.isContinuous()){
        cols = static_cast<int>(input.total() * input.channels());
        rows = 1;
    }

    for(int row = 0; row != rows; ++row){
        auto begin = input.template ptr<T>(row);
        auto end = begin + cols;
        while(begin != end){
            func(*begin);
            ++begin;
        }
    }

    return func;
}

template<typename T, typename BinaryFunc, typename MatA, typename MatB>
BinaryFunc for_each_channels(MatA &&input_1, MatB &&input_2, BinaryFunc func)
{    
    CV_Assert(input_1.rows == input_2.rows &&
              input_1.cols == input_2.cols &&
              input_1.type() == input_2.type());

    int rows = input_1.rows;
    int cols = input_1.cols;

    if(input_1.isContinuous() && input_2.isContinuous()){
        rows = 1;
        cols = static_cast<int>(input_1.total() * input_1.channels());
    }

    for(int row = 0; row != rows; ++row){
        auto input_1_begin = input_1.template ptr<T>(row);
        auto input_2_begin = input_2.template ptr<T>(row);
        auto input_1_end = input_1_begin + cols;
        while(input_1_begin != input_1_end){
            func(*input_1_begin, *input_2_begin);
            ++input_1_begin; ++input_2_begin;
        }
    }

    return func;
}

/**
 *@brief apply for_each algorithm on variadic channels.\n
 * This function is similar to for_each_channels, but more\n
 * flexible
 *
 *@param inout : input and output
 *@param func : the functor accept atleast 1 arguments,\n
 * the first argument type must be Mat
 *@param T : template parameter, channel type of src
 *@param args : variadic parameters which forward to the func
 */

/*! \brief example.
 *\code
 * cv::Mat_<cv::Vec3b> input =
 * cv::imread("give_me_back_my_money.jpg");
 *
 * for_each<cv::Vec3b>(input, [](cv::vec3b &a)
 * {
 *     a[0] = 255 - a[0];
 *     a[1] = 255 - a[1];
 *     a[2] = 255 - a[2];
 * });
 *\endcode
*/
template<typename T, typename Mat, typename Func,
         typename... Args>
Func for_each(Mat &&inout, Func func, Args... args)
{
    int rows = inout.rows;
    int cols = inout.cols;

    if(inout.isContinuous()){
        cols = static_cast<int>(inout.total());
        rows = 1;
    }

    for(int row = 0; row != rows; ++row){
        auto begin = inout.template ptr<T>(row);
        auto end = begin + cols;
        while(begin != end){
            func(*begin, std::forward<Args>(args)...);
            ++begin;
        }
    }

    return func;
}

/**
 *@brief apply for_each algorithm on variadic channels.\n
 * This function is similar to for_each_channels, but more\n
 * flexible
 *
 *@param inout_1 : input and output 1
 *@param inout_2 : input and output 2
 *@param func : the functor accept atleast 2 arguments,\n
 * the first and second argument type must be Mat
 *@param T : template parameter, channel type of src
 *@param args : variadic parameters which forward to the func
 */

/*! \brief example.
 *\code
 * cv::Mat_<cv::Vec3b> im_1 =
 * cv::imread("give_me_back_my_money.jpg");
 * cv::Mat_<cv::Vec3b> im_2 =
 * cv::imread("charlotte.jpg");
 *
 * for_each<cv::Vec3b>(input, [](cv::vec3b &a, cv::vec3b &b)
 * {
 *     b[0] = 255 - a[0];
 *     b[1] = 255 - a[1];
 *     b[2] = 255 - a[2];
 * });
 *\endcode
*/
template<typename T, typename MatA, typename MatB, typename Func,
         typename... Args>
Func for_each(MatA &&inout_1, MatB &&inout_2,
              Func func, Args... args)
{
    CV_Assert(inout_1.total() == inout_2.total());
    CV_Assert(inout_1.type() == inout_2.type());

    int rows = inout_1.rows;
    int cols = inout_1.cols;

    if(inout_1.isContinuous()){
        cols = static_cast<int>(inout_1.total());
        rows = 1;
    }

    for(int row = 0; row != rows; ++row){
        auto begin_a = inout_1.template ptr<T>(row);
        auto begin_b = inout_2.template ptr<T>(row);
        auto end = begin_a + cols;
        while(begin_a != end){
            func(*begin_a, *begin_b,
                 std::forward<Args>(args)...);
            ++begin_a;
        }
    }

    return func;
}

} /*! @} End of Doxygen Groups*/

#endif // GENERICFOREACH_HPP
