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
 *@param seed seed of random engine
 *@warning The range of beg2 must not smaller than beg1
 */
template<typename ForwardIter1, typename ForwardIter2,
         typename URNG>
void shuffles(ForwardIter1 beg1, ForwardIter1 end1,
              ForwardIter2 beg2, URNG &&g)
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

    std::shuffle(std::begin(key_values), std::end(key_values), g);
    for(auto it = std::begin(key_values);
        it != std::end(key_values); ++it, ++beg1, ++beg2){
        *beg1 = std::move(it->first);
        *beg2 = std::move(it->second);
    }
}

template<typename ForwardIter1, typename ForwardIter2,
         typename URNG>
void shuffles(ForwardIter1 beg1, ForwardIter1 end1,
              ForwardIter2 beg2, unsigned int seed)
{
    shuffles(beg1, end1, beg2,
             std::default_random_engine(seed));
}

/**
 * Overload version of shuffles, accept containers
 * rather than iterator
 */
template<typename T, typename U, typename URNG>
inline
void shuffles(T &fir, U &sec, URNG &&g)
{
    shuffles(std::begin(fir), std::end(fir),
             std::begin(sec), g);
}

template<typename T, typename U>
inline
void shuffles(T &fir, U &sec, unsigned int seed = 0)
{
    shuffles(std::begin(fir), std::end(fir),
             std::begin(sec), std::default_random_engine(seed));
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/


#endif // OCV_SHUFFLE_DATA_HPP
