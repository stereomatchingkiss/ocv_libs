#ifndef OCV_SPLIT_TRAIN_TEST_HPP
#define OCV_SPLIT_TRAIN_TEST_HPP

/*! \file split_train_test.hpp
    \brief split input data to two sets of data
*/

#include <opencv2/core.hpp>

#ifdef OCV_SUPPORT_DLIB
#include <dlib/image_processing.h>
#endif

#include <random>
#include <tuple>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup ml
 *  @{
 */
namespace ml{

namespace details{

inline
void swap(cv::Mat &lhs, cv::Mat &rhs)
{
    cv::swap(lhs, rhs);
}

template<typename T>
inline
void swap(cv::Mat_<T> &lhs, cv::Mat_<T> &rhs)
{
    cv::swap(lhs, rhs);
}

#ifdef OCV_SUPPORT_DLIB
template<typename T>
inline
void swap(dlib::array2d<T> &lhs, dlib::array2d<T> &rhs)
{
    lhs.swap(rhs);
}
#endif

template<typename T>
inline
void swap(T &lhs, T &rhs)
{
    std::swap(lhs, rhs);
}

}

/**
 * split input data and input label to two sets of data, this function
 * may change the content of input_data and input_label if they can be
 * swapped, this way it could avoid expensive copy operation
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @param train_data training data split from input data
 * @param train_label training label split from input data
 * @param test_data test data split from input data
 * @param test_label test label split from input data
 * @param test_ratio determine the ratio of the input data
 * split to test data
 * @param shuffle if true, shuffle the train data and label;
 * else do not shuffle.Default value is true
 */
template<typename Data, typename Label>
void
split_train_test_inplace(Data &input_data, Label &input_label,
                         Data &train_data, Label &train_label,
                         Data &test_data, Label &test_label,
                         double test_ratio,
                         bool shuffle = true)
{
    size_t const train_size = input_data.size() -
            static_cast<size_t>(input_data.size() * test_ratio);
    enum class tag : unsigned char{
        test_tag,
        train_tag,
    };
    std::vector<tag> seed(input_data.size(), tag::train_tag);    
    for(size_t i = train_size; i != input_data.size(); ++i){
        seed[i] = tag::test_tag;        
    }    
    if(shuffle){
        std::random_device rd;
        std::default_random_engine g(rd());
        std::shuffle(std::begin(seed), std::end(seed), g);
    }

    size_t const test_size = input_data.size() - train_size;
    test_data.resize(test_size);
    test_label.resize(test_size);
    train_data.resize(train_size);
    train_label.resize(train_size);
    size_t test_index = 0;
    size_t train_index = 0;
    for(size_t i = 0; i != seed.size(); ++i){
        if(seed[i] == tag::train_tag){
            details::swap(train_label[train_index], input_label[i]);
            details::swap(train_data[train_index], input_data[i]);
            ++train_index;
        }else{
            details::swap(test_label[test_index], input_label[i]);
            details::swap(test_data[test_index], input_data[i]);
            ++test_index;
        }
    }
}

/**
 * split input data and input label to two sets of data, this function
 * may change the content of input_data and input_label if they can be
 * swapped, this way it could avoid expensive copy operation
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @param test_ratio determine the ratio of the input data
 * split to test data
 * @return train_data, train_label, test_data, test_label
 * @param shuffle if true, shuffle the train data and label;
 * else do not shuffle.Default value is true
 * @warning cannot compile if type Data and Label are not copy constructible
 * or move constructible
 */
template<typename Data, typename Label>
std::tuple<Data, Label, Data, Label>
split_train_test_inplace(Data &input_data, Label &input_label,
                         double test_ratio,
                         bool shuffle = true)
{
    static_assert((std::is_copy_constructible<Data>::value &&
                   std::is_copy_constructible<Label>::value) ||
                  (std::is_move_constructible<Data>::value &&
                   std::is_move_constructible<Label>::value),
                  "type Data and type Label must be copy constructible or "
                  "move constructible");

    Data test_data;
    Label test_label;
    Data train_data;
    Label train_label;

    split_train_test_inplace(input_data, input_label, train_data, train_label,
                             test_data, test_label, test_ratio, shuffle);

    return std::make_tuple(train_data, train_label,
                           test_data, test_label);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // SPLIT_TRAIN_TEST_HPP

