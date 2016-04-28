#ifndef OCV_ARMA_DIST_METRIC_HPP
#define OCV_ARMA_DIST_METRIC_HPP

#include "../arma/type_traits.hpp"

#include <armadillo>

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

namespace details{

template<typename T,
         typename Hist1,
         typename Hist2,
         typename Index>
struct type_constraint
{
    static_assert(std::is_floating_point<T>::value,
                  "T should be floating point");
    static_assert(std::is_same<typename Hist1::elem_type,
                  typename Hist2::elem_type>::value,
                  "elem type of Hist1 and Hist2 "
                  "should be the same one");
    static_assert(std::is_floating_point<typename Hist1::elem_type>::value,
                  "elem type of Hist1 and Hist2 should "
                  "be floating point");
    static_assert(std::is_integral<Index>::value,
                  "Index should be integral");
    static_assert(is_two_dim<Hist2>::value,
                  "Hist2 should be arma::Mat or "
                  "arma::SpMat");
};

template<typename T, typename U>
inline
typename std::enable_if<
arma::is_arma_sparse_type<U>::value ||
(arma::is_arma_type<U>::value &&
 !std::is_same<T, typename U::elem_type>::value),
arma::Col<T>>::type
to_colvec(U const &input)
{    
    return arma::Col<T>(input);
}

template<typename T, typename U>
inline
typename std::enable_if<
arma::is_arma_type<U>::value &&
std::is_same<T, typename U::elem_type>::value,
U>::type
to_colvec(U const &input)
{    
    return input;
}

}

/**
 * measure chi square distance
 * @tparam T return type of compare
 */
template<typename T = float>
struct cosine_similarity
{
    using result_type = T;

    template<typename Hist1,
             typename Hist2,
             typename Index>
    T compare(Hist1 const &query_hist,
              Hist2 const &datahist,
              Index const &index) const
    {
        using namespace details;
        details::type_constraint<T,Hist1,Hist2,Index>();

        return similarity_compute(to_colvec<T>(query_hist),
                                  to_colvec<T>(datahist.col(index)));
    }

private:    
    template<typename U, typename V>
    T similarity_compute(U const &lhs,
                         V const &rhs) const
    {
        auto const denom =
                std::sqrt(arma::sum(lhs % lhs)) *
                std::sqrt(arma::sum(rhs % rhs)) +
                T(1e-10);

        return arma::sum((lhs % rhs)) / (denom);
    }
};

/**
 * measure chi square distance
 * @tparam T return type of compare
 */
template<typename T = float>
struct chi_square{

    using result_type = T;

    template<typename Hist1,
             typename Hist2,
             typename Index>
    T compare(Hist1 const &query_hist,
              Hist2 const &datahist,
              Index const &index) const
    {
        using namespace details;
        type_constraint<T,Hist1,Hist2,Index>();

        return chi_square_compute(to_colvec<T>(query_hist),
                                  to_colvec<T>(datahist.col(index)));
    }

private:    
    template<typename U, typename V>
    T chi_square_compute(U const &lhs,
                         V const &rhs) const
    {
        return arma::sum(arma::square(lhs - rhs) /
                         (lhs + rhs + T(1e-10)));
    }

};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // DIST_METRIC_HPP
