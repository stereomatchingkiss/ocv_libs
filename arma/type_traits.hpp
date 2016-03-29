#ifndef OCV_ARMA_TYPE_TRAITS_HPP
#define OCV_ARMA_TYPE_TRAITS_HPP

#include <armadillo>

#include <type_traits>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup armd
 *  @{
 */
namespace armd{

template<typename T>
struct is_spmat
{
    using type = typename std::is_same<T,
    arma::SpMat<typename T::elem_type>>::type;

    enum{value = std::is_same<type, std::true_type>::value};
};

template<typename T>
struct is_mat
{
    using type = typename std::is_same<T,
    arma::Mat<typename T::elem_type>>::type;

    enum{value = std::is_same<type, std::true_type>::value};
};

template<typename T>
struct is_two_dim
{
    using type  = typename std::conditional<is_spmat<T>::value ||
    is_mat<T>::value, std::true_type, std::false_type>::type;

    enum{value = std::is_same<type, std::true_type>::value};
};

template<typename T>
struct is_col
{
    using type = typename std::is_same<T,
    arma::Col<typename T::elem_type>>::type;

    enum{value = std::is_same<type, std::true_type>::value};
};

template<typename T>
struct is_row
{
    using type = typename std::is_same<T,
    arma::Row<typename T::elem_type>>::type;

    enum{value = std::is_same<type, std::true_type>::value};
};

template<typename T>
struct is_arma_matrix
{
    enum {value = is_spmat<T>::value || is_mat<T>::value ||
         is_row<T>::value || is_col<T>::value};
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_ARMA_TYPE_TRAITS_HPP
