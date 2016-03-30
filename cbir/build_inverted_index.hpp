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

template<typename Key, typename Value, typename Vocab>
void build_inverted_index(Vocab const &vocab,
                          inverted_index<Key, Value> &index)
{
    static_assert(armd::is_two_dim<Vocab>::value,
                  "Vocab should be arma::Mat or "
                  "arma::SpMat");

    index.clear();
    for(arma::uword i = 0; i != vocab.n_cols; ++i){
        auto const &vcol = vocab.col(i);
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
