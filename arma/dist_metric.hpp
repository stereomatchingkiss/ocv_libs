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

/**
 * measure chi square distance
 */
struct chi_square{

    using result_type = double;

    template<typename Hist1,
             typename Hist2,
             typename Index>
    result_type compare(Hist1 const &query_hist,
                        Hist2 const &datahist,
                        Index const &index) const
    {
        static_assert(std::is_same<typename Hist1::elem_type,
                      typename Hist2::elem_type>::value,
                      "elem type of Hist1 and Hist2 "
                      "should be the same one");
        static_assert(std::is_integral<Index>::value,
                      "Index should be integral");
        static_assert(is_two_dim<Hist2>::value,
                      "Hist2 should be arma::Mat or "
                      "arma::SpMat");

        return compare_impl(query_hist, datahist, index);
    }

private:
    template<typename T>
    double chi_square_compute(arma::Col<T> const &lhs,
                              arma::Col<T> const &rhs) const
    {
        return 0.5 * arma::sum(arma::square(lhs - rhs) /
                               (lhs + rhs + 1e-10));
    }

    template<typename Hist1,
             typename Hist2,
             typename Index>
    typename std::enable_if<
    !std::is_same<typename Hist1::elem_type, double>::value,
    double>::type
    compare_impl(Hist1 const &query_hist,
                 Hist2 const &datahist,
                 Index const &index) const
    {
        using colvec = arma::Col<double>;

        auto const &dhist_view = datahist.col(index);
        colvec qhist(query_hist.n_elem);
        colvec dhist(dhist_view.n_elem);
        for(arma::uword i = 0; i != query_hist.n_elem; ++i){
            qhist(i) = query_hist(i);
            dhist(i) = dhist_view(i);
        }

        return chi_square_compute(qhist, dhist);
    }

    template<typename Hist1,
             typename Hist2,
             typename Index>
    typename std::enable_if<
    std::is_same<typename Hist1::elem_type, double>::value,
    double>::type
    compare_impl(Hist1 const &query_hist,
                 Hist2 const &datahist,
                 Index const &index) const
    {
        using colvec = arma::Col<double>;

        auto const &dhist_view = datahist.col(index);
        colvec qhist(query_hist.memptr(), query_hist.n_elem);
        colvec dhist(dhist_view.memptr(), dhist_view.n_elem);

        return chi_square_compute(qhist, dhist);
    }

};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // DIST_METRIC_HPP
