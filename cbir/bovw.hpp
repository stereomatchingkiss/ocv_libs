#ifndef OCV_CBIR_BOVW_HPP
#define OCV_CBIR_BOVW_HPP

#include "../arma/type_traits.hpp"

#include <opencv2/core.hpp>

#include <armadillo>

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
 *Convert the features of images into histogram
 *@tparam T type of the code_book and features
 *@tparam Hist type of histogram
 *@code
 *arma::Mat<float> code_book;
 *code_book.load("ukbench.h5", arma::hdf5_binary);
 *bovw<float> bv(std::move(code_book));
 *arma::Mat<float> features;
 *features.load("uk3000", arma::raw_ascii);
 *auto const hist = bv.describe(features);
 *@endcode
 */
template<typename T, typename Hist = arma::SpMat<arma::uword>>
class bovw
{    
public:
    bovw()
    {
        static_assert(std::is_arithmetic<T>::value,
                      "T should be arithmetic type");

        static_assert(armd::is_arma_matrix<Hist>::value,
                      "Hist should be arma::SpMat, arma::Mat,"
                      "arma::Col or arma::Row");
    }

    /**
     * Convert the features of images into histogram
     * @param features features of the image
     * @param code_book code book of the data sets
     * @return histogram of bovw
     */
    Hist describe(arma::Mat<T> const &features,
                  arma::Mat<T> const &code_book) const
    {
        arma::Mat<T> dist(features.n_cols, code_book.n_cols);
        for(arma::uword i = 0; i != features.n_cols; ++i){
            dist.row(i) = euclidean_dist(features.col(i),
                                         code_book);
        }
        //dist.print("dist");

        Hist hist = create_hist(dist.n_cols,
                                armd::is_two_dim<Hist>::type());
        for(arma::uword i = 0; i != dist.n_rows; ++i){
            arma::uword min_idx;
            dist.row(i).min(min_idx);
            ++hist(min_idx);
        }
        //hist.print("\nhist");

        return hist;
    }

private:    
    Hist create_hist(arma::uword size, std::true_type) const
    {
        return arma::zeros<Hist>(size,1);
    }

    Hist create_hist(arma::uword size, std::false_type) const
    {
        return arma::zeros<Hist>(size);
    }

    arma::Mat<T> euclidean_dist(arma::Col<T> const &x,
                                arma::Mat<T> const &y) const
    {
        return arma::sqrt(arma::sum
                          (arma::square(y.each_col() - x)));
    }
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_BOVW_HPP
