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

    template<typename Hist1,
             typename Hist2,
             typename Index>
    double compare(Hist1 const &query_hist,
                   Hist2 const &datahist,
                   Index const &index) const
    {
        static_assert(std::is_integral<Index>::value,
                      "Index should be integral");
        static_assert(is_two_dim<Hist2>::value,
                      "Hist2 should be arma::Mat or "
                      "arma::SpMat");

        using colvec = arma::Col<double>;

        auto const &dhist_view = datahist.col(index);
        colvec qhist(query_hist.n_elem);
        colvec dhist(dhist_view.n_elem);
        for(arma::uword i = 0; i != query_hist.n_elem; ++i){
            qhist(i) = query_hist(i);
            dhist(i) = dhist_view(i);
        }

        return 0.5 * arma::sum(arma::square(qhist - dhist) /
                (qhist + dhist + 1e-10));
    }

};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // DIST_METRIC_HPP
