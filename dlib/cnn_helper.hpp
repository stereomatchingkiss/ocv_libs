#ifndef OCV_ODLIB_CNN_HELPER_HPP
#define OCV_ODLIB_CNN_HELPER_HPP

#include <dlib/dnn.h>

#include <type_traits>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup odlib
 *  @{
 */
namespace odlib{

template<typename Net>
inline
decltype(std::declval<Net>().layer_details().set_learning_rate_multiplier(0))
set_learning_rate_impl(Net &net, double rate)
{
    net.layer_details().set_learning_rate_multiplier(rate);
    net.layer_details().set_bias_learning_rate_multiplier(rate);
}

inline
void set_learning_rate_impl(...)
{
}

template<int from, int to, typename Net>
typename std::enable_if<from == to>::type
set_learning_rate(Net&, double)
{
}

template<int from, int to, typename Net>
typename std::enable_if<from != to>::type
        set_learning_rate(Net &net, double rate)
{
        set_learning_rate_impl(dlib::layer<from>(net), rate);
        set_learning_rate<from + 1, to>(net, rate);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // CNN_HELPER_HPP
