#ifndef OCV_CBIR_BUILD_INVERTED_INDEX_HPP
#define OCV_CBIR_BUILD_INVERTED_INDEX_HPP

#include "../arma/type_traits.hpp"
#include "../core/inverted_index.hpp"

#include <cmath>

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
 *@param hist histogram of the dataset,each col associate to
 *one histogram
 *@param index inverted index associate with the histogram
 *@code
 *arma::SpMat<arma::uword> hist;
 *vocab.load("ukbench_hist");
 *ocv::inverted_index<arma::uword, arma::uword> invert;
 *ocv::cbir::build_inverted_index(hist, invert);
 * //The result of invert may looks like
 * //0 : 1, 45, 77, 1029
 * //1 : 33, 64, 88
 * //....................
 * //left side is the features(key), right side is the
 * //documents(value) the features(key) appear
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

/**
 *convert inverted index to inverse document frequency
 *@param index invert index associate with the histogram
 *@param sample_size the size of the sample
 *@param idf inverse document frequency
 *@code
 *arma::SpMat<arma::uword> hist;
 *vocab.load("ukbench_hist");
 *using namespace ocv::cbir;
 *ocv::inverted_index<arma::uword, arma::uword> invert;
 *build_inverted_index(hist, invert);
 *std::map<arma::uword, double> idf;
 *convert_to_idf(invert, hist.n_cols, idf);
 *@endcode
 */
template<typename Key, typename Value, typename IDF>
void convert_to_idf(inverted_index<Key, Value> const &index,
                    size_t sample_size,
                    IDF &idf)
{
    using key_type = typename IDF::key_type;
    using mapped_type = typename IDF::mapped_type;

    static_assert(std::is_class<IDF>::value,
                  "IDF should be a class");
    static_assert(std::is_floating_point<mapped_type>::value,
                  "mapped_type of IDF should be floating point");
    static_assert(std::is_same<Key, key_type>::value,
                  "key_type of IDF should be the same as Key");

    idf.clear();
    auto const total = static_cast<mapped_type>(sample_size);
    for(auto const &val : index){
        auto const fre_size =
                static_cast<mapped_type>(val.second.size());
        idf.insert({val.first,
                    std::max(std::log(total/(fre_size+1)), mapped_type(0))});
    }
}


} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_BUILD_INVERTED_INDEX_HPP
