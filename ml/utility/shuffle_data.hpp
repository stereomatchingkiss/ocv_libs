#ifndef OCV_SHUFFLE_DATA_HPP
#define OCV_SHUFFLE_DATA_HPP

#include <iterator>
#include <random>
#include <type_traits>
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

/**
 *Shuffle key value pair
 *
 *@param beg1 the begin position of first container
 *@param end1 the end position of first container
 *@param beg2 the begin position of second container
 *@warning The range of beg2 must not smaller than beg1
 */
template<typename ForwardIter1, typename ForwardIter2>
void shuffles(ForwardIter1 beg1, ForwardIter1 end1,
              ForwardIter2 beg2)
{
    using vtype1 = std::decay<decltype(*beg1)>::type;
    using vtype2 = std::decay<decltype(*beg2)>::type;
    using value_type = std::pair<vtype1, vtype2>;

    std::vector<value_type> key_values;
    ForwardIter1 beg1_cpy = beg1;
    ForwardIter2 beg2_cpy = beg2;
    for(; beg1_cpy != end1; ++beg1_cpy, ++beg2_cpy){
        key_values.emplace_back(std::move(*beg1_cpy),
                                std::move(*beg2_cpy));
    }

    std::random_device rd;
    std::default_random_engine g(rd());
    std::shuffle(std::begin(key_values), std::end(key_values), g);
    for(auto it = std::begin(key_values);
        it != std::end(key_values); ++it, ++beg1, ++beg2){
        *beg1 = std::move(it->first);
        *beg2 = std::move(it->second);
    }
}

/**
 * Overload version of shuffles, accept containers
 * rather than iterator
 */
template<typename T, typename U>
inline
void shuffles(T &fir, U &sec)
{
    shuffles(std::begin(fir), std::end(fir),
             std::begin(sec));
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/


#endif // OCV_SHUFFLE_DATA_HPP
