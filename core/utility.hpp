#ifndef OCV_CORE_UTILITY_HPP
#define OCV_CORE_UTILITY_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <opencv2/core/core.hpp>

#include "for_each.hpp"

namespace ocv
{

/*
 *@brief easy function for compare_channels, user should make sure T is the correct
 *channel type of src1.
 */
template<typename T, typename UnaryFunc>
typename std::enable_if<!std::is_same<UnaryFunc, cv::Mat>::value, bool>::type
compare_channels(cv::Mat const &src, UnaryFunc func)
{
    int rows = src.rows;
    int cols = src.cols * src.channels();

    if(src.isContinuous()){
        rows = 1;
        cols = src.total() * src.channels();
    }

    for(int row = 0; row != rows; ++row){
        T const *src_ptr = src.ptr<T>(row);
        for(int col = 0; col != cols; ++col){
            if(!func(src_ptr[col])){
                return false;
            }
        }
    }

    return true;
}

/**
 *@brief : easy function for compare_channels, user should make sure T is the correct
 *channel type of src1 and src2
 */
template<typename T, typename BiFunc = std::equal_to<T> >
bool compare_channels(cv::Mat const &src1, cv::Mat const &src2, BiFunc func = std::equal_to<T>())
{
    if(src1.rows != src2.rows || src1.cols != src2.cols || src1.type() != src2.type()){
        return false;
    }

    if(src1.isContinuous() && src2.isContinuous()){
        return std::equal(src1.ptr<T>(0), src1.ptr<T>(0) + src1.total() * src1.channels(), src2.ptr<T>(0), func);
    }

    int const rows = src1.rows;
    int const pixels_per_row = src1.cols * src1.channels();
    for(int row = 0; row != rows; ++row){
        T const *src1_ptr = src1.ptr<T>(row);
        T const *src2_ptr = src2.ptr<T>(row);
        for(int col = 0; col != pixels_per_row; ++col){
            if(!func(src1_ptr[col], src2_ptr[col])){
                return false;
            }
        }
    }

    return true;
}


/**
 * @brief: copy src to dst if their rows, cols or type are different
 */
inline 
void copy_if_not_same(cv::Mat const &src, cv::Mat &dst)
{
    if(src.data != dst.data){
        src.copyTo(dst);
    }
}

inline 
void create_mat(cv::Mat const &src, cv::Mat &dst)
{
    dst.create(src.rows, src.cols, src.type());
}

/**
 * @brief : experimental version for cv::Mat, try to alleviate the problem
 * of code bloat.User should make sure the space of begin point to
 * have enough of spaces.
 */
template<typename T, typename InputIter>
void copy_to_one_dim_array(cv::Mat const &src, InputIter begin)
{       
    if(src.isContinuous()){
        auto ptr = src.ptr<T>(0);
        std::copy(ptr, ptr + src.total() * src.channels(), begin);
        return;
    }

    size_t const pixel_per_row = src.cols * src.channels();
    for(int row = 0; row != src.rows; ++row){
        auto ptr = src.ptr<T>(row);
        std::copy(ptr, ptr + pixel_per_row, begin);
        begin += pixel_per_row;
    }
}

template<typename T>
std::vector<T> copy_to_one_dim_array(cv::Mat const &src)
{
    std::vector<T> result(src.total() * src.channels());
    copy_to_one_dim_array<T>(src, std::begin(result));

    return result;
}

/**
 * @brief experimental version for cv::Mat, try to alleviate the problem
 * of code bloat.User should make sure the space of begin point to
 * have enough of spaces.
 */
template<typename T, typename InputIter>
void copy_to_one_dim_array_ch(cv::Mat const &src, InputIter begin, int channel)
{
    int const channel_number = src.channels();
    if(channel_number <= channel || channel < 0){
        throw std::runtime_error("channel value is invalid\n" + std::string(__FUNCTION__) +
                                 "\n" + std::string(__FILE__));
    }

    for(int row = 0; row != src.rows; ++row){
        auto ptr = src.ptr<T>(row) + channel;
        for(int col = 0; col != src.cols; ++col){
            *begin = *ptr;
            ++begin;
            ptr += channel_number;
        }
    }
}

template<typename T>
std::vector<T> copy_to_one_dim_array_ch(cv::Mat const &src, int channel)
{
    std::vector<T> result(src.total());
    copy_to_one_dim_array_ch<T>(src, std::begin(result), channel);

    return result;
}

/**
 * Expand the region.
 * ex : x1=30,x2=60,y1=30,y2=60, after expand 16pix,
 * result would be x1=14, x2=76, y1=14,y2=76.If the
 * region out of range, the region will be clipped
 * @param img_size size of the image
 * @param region region within the image
 * @param expand_pix how many pixel want to expand
 * @return region after expand
 */
inline
cv::Rect expand_region(cv::Size const &img_size,
                       cv::Rect const &region,
                       int expand_pix)
{
    cv::Rect result = region;
    result.x = std::max(0, result.x - expand_pix);
    result.width = std::min(result.width + 2*expand_pix,
                          img_size.width - result.x - 1);

    result.y = std::max(0, result.y - expand_pix);
    result.height = std::min(result.height + 2*expand_pix,
                           img_size.height - result.y - 1);
    return result;
}

template<typename T,
         typename Distribution =
         typename std::conditional<std::is_integral<T>::value,
                                   std::uniform_int_distribution<T>,
                                   std::uniform_real_distribution<T>>::type
         >
void generate_random_value(cv::Mat &inout, T epsillon,
                           Distribution &&distribution = Distribution(0, 100))
{
    static_assert(std::is_arithmetic<T>::value, "T should be arithmetic type");

    std::random_device rd;
    std::default_random_engine re(rd());

    for_each_channels<T>(inout, [&](T &value)
    {
        value = distribution(re);
    });

    inout *= (2 * epsillon);
    inout -= epsillon;
}

}

#endif // OCV_CORE_UTILITY_HPP
