#ifndef OCV_CBIR_BUILD_INVERTED_INDEX_HPP
#define OCV_CBIR_BUILD_INVERTED_INDEX_HPP

#include "../arma/type_traits.hpp"
#include "../core/inverted_index.hpp"

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup cbir
 *  @{
 */
namespace cbir{

/**
 *build inverted index based on the histograms
 *@param hist histogram of the dataset
 *@param index invert index associate with the histogram
 *@code
 *arma::SpMat<arma::uword> hist;
 *vocab.load("ukbench_hist");
 *ocv::inverted_index<arma::uword, arma::uword> invert;
 *ocv::cbir::build_inverted_index(hist, invert);
 *@endcode
 */
template<typename Key, typename Value, typename Hist>
void build_inverted_index(Hist const &hist,
                          inverted_index<Key, Value> &index)
{
    static_assert(armd::is_two_dim<Hist>::value,
                  "Vocab should be arma::Mat or "
                  "arma::SpMat");

    index.clear();
    for(arma::uword i = 0; i != hist.n_cols; ++i){
        auto const &vcol = hist.col(i);
        for(auto it = vcol.begin();
            it != vcol.end(); ++it){
            if(*it != 0){
                index.insert(it.row(), i);
            }
        }
    }
}


} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_BUILD_INVERTED_INDEX_HPP
