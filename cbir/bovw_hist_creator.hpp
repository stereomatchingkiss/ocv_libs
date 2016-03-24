#ifndef OCV_CBIR_BOVW_HIST_CREATOR_HPP
#define OCV_CBIR_BOVW_HIST_CREATOR_HPP

#include "bovw.hpp"
#include "features_indexer.hpp"

#include <opencv2/core.hpp>

#include <armadillo>

#include <omp.h>
#include <mutex>

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
 *Create histogram of the images associate with the codebook
 *@tparam CodeBook type of the codebook
 *@tparam Feature type of the feature
 *@tparam Hist type of the histogram
 *@code
 *features_indexer fi("ukbench.h5");
 *arma::Mat<float> code_book;
 *code_book.load("ukbench_code_book", arma::arma_ascii);
 *
 *bovw_hist_creator<float, float> bh(fi);
 *auto const hist = bh.create(code_book);
 *hist.save("ukbench_hist");
 *@endcode
 */
template<typename CodeBook, typename Feature,
         typename Hist = arma::SpMat<CodeBook>>
class bovw_hist_creator
{
public:
    explicit bovw_hist_creator(features_indexer const &fi) :
        fi_(fi)
    {
        static_assert(std::is_arithmetic<CodeBook>::value,
                      "CodeBook should be arithmetic type");
        static_assert(std::is_arithmetic<Feature>::value,
                      "Feature should be arithmetic type");

        enum {is_spmat = std::is_same<Hist, arma::SpMat<CodeBook>>::value};
        enum {is_mat = std::is_same<Hist, arma::Mat<CodeBook>>::value};

        static_assert(is_spmat || is_mat,
                      "Hist should be arma::Mat or "
                      "arma::SpMat");
    }

    /**
     * Create the histogram of each image associated with
     * the vocab(code book)
     * @param vocab the code book(vocab)
     * @return histogram of each image(one col per image)
     */
    Hist create(arma::Mat<CodeBook> const &vocab) const
    {
        auto const img_size =
                (fi_.get_names_dimension()[0]);
        Hist hist(vocab.n_cols, img_size);
        bovw<CodeBook, Hist> bv;

#pragma omp parallel for
        for(int i = 0; i < img_size; ++i){
            cv::Mat features;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                fi_.read_image_features(features, i);
            }
            arma::Mat<CodeBook> const vocab_features =
                    to_arma(features,
                            typename std::is_same<CodeBook, Feature>::type());
            auto const shist = bv.describe(vocab_features, vocab);
            {
                std::lock_guard<std::mutex> guard(mutex_);
                hist.col(i) = shist;
            }
        }

        return hist;
    }

private:
    arma::Mat<CodeBook>
    to_arma(cv::Mat &features, std::true_type) const
    {
        return arma::Mat<CodeBook>(features.ptr<Feature>(0),
                                   features.cols,
                                   features.rows,
                                   false);
    }

    arma::Mat<CodeBook>
    to_arma(cv::Mat const &features, std::false_type) const
    {
        arma::Mat<CodeBook> result(features.cols, features.rows);
        //do not use std::copy because it may generate warning
        auto *ptr = features.ptr<Feature>(0);
        arma::uword const total = features.cols * features.rows;
        for(arma::uword i = 0; i != total; ++i){
            result(i) = static_cast<CodeBook>(ptr[i]);
        }

        return result;
    }

    features_indexer const &fi_;
    mutable std::mutex mutex_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_BOVW_HIST_CREATOR_HPP
