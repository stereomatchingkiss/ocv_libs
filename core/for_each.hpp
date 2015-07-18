#ifndef GENERICFOREACH_HPP
#define GENERICFOREACH_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

namespace ocv
{

namespace details
{

template<typename T>
void unpack_unary_funcs(T)
{
}

template<typename T, typename UnaryFunc, typename... UnaryFuncs>
void unpack_unary_funcs(T value_ptr, UnaryFunc func, UnaryFuncs... funcs)
{
    func(*value_ptr);
    UnaryFunc(++value_ptr, funcs);
}

}

template<typename T, typename UnaryFunc, typename Mat>
UnaryFunc for_each_channels(Mat &&input, UnaryFunc func);

/**
 *@brief process the each block of the image
 *@param input the input image
 *@param block the size of the subimage
 *@param func functor accept three parameters.\n
 * col : x-axis position of the image(not block)\n
 * row : y-axis position of the image(not block)\n
 * subimage : the subimage of image
 *@param stride the step when sliding the subimage
 */
template<typename TriFunc>
TriFunc
for_each_block(cv::Mat const &input,
               cv::Size2i const &block,
               TriFunc func,
               cv::Size2i const &stride = {1, 1})
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
        cols = input.total() * input.channels();
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

template<typename T, typename BinaryFunc, typename Mat>
BinaryFunc for_each_channels(Mat &&input_1, Mat &&input_2, BinaryFunc func)
{    
    CV_Assert(input_1.rows == input_2.rows && input_1.cols == input_2.cols && input_1.type() == input_2.type());

    int rows = input_1.rows;
    int cols = input_1.cols;

    if(input_1.isContinuous() && input_2.isContinuous()){
        rows = 1;
        cols = input_1.total() * input_1.channels();
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
 *@brief apply for_each algorithm on variadic channels
 *
 *@param inout : input and output
 *@param funcs : unary functor packs,
 * each func handle one channel element
 *@param T : template parameter, channel type of src
 */
template<typename T, typename Mat, typename... UnaryFuncs>
void for_each_variadic_channels(Mat &&inout, UnaryFuncs... funcs)
{   
    CV_Assert(inout.channels() == sizeof...(funcs));

    int rows = inout.rows;
    int cols = inout.cols;
    if(inout.isContinuous()){
        cols = inout.total() * sizeof...(funcs);
        rows = 1;
    }

    for(int row = 0; row != rows; ++row){
        auto dst_ptr_begin = inout.template ptr<T>(row);
        auto dst_ptr_end = dst_ptr_begin + cols;
        for(; dst_ptr_begin != dst_ptr_end; dst_ptr_begin += sizeof...(funcs)){
            details::unpack_unary_funcs(dst_ptr_begin, funcs);
            //func_one(*dst_ptr_begin); //++dst_ptr;
            //func_two(dst_ptr_begin[1]); //++dst_ptr;
            //func_three(dst_ptr_begin[2]); //++dst_ptr;
        }
    }
}

}

#endif // GENERICFOREACH_HPP
