#ifndef OCV_CBIR_BOVW_HPP
#define OCV_CBIR_BOVW_HPP

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
    explicit bovw(arma::Mat<T> code_book) :
        code_book_(std::move(code_book))
    {
        static_assert(std::is_arithmetic<T>::value,
                      "T should be arithmetic type");

        static_assert(is_arma::value,
                      "Hist should be arma::SpMat, arma::Mat,"
                      "arma::Col or arma::Row");
    }

    /**
     * Convert the features of images into histogram
     * @param features features of the image
     * @return histogram of bovw
     */
    Hist describe(arma::Mat<T> &features) const
    {
        arma::Mat<T> dist(features.n_cols, code_book_.n_cols);
        for(arma::uword i = 0; i != features.n_cols; ++i){
            dist.row(i) = euclidean_dist(features.col(i),
                                         code_book_);
        }
        //dist.print("dist");

        Hist hist = create_hist(dist.n_cols,
                                is_two_dim::type());
        for(arma::uword i = 0; i != dist.n_rows; ++i){
            arma::uword min_idx;
            dist.row(i).min(min_idx);
            ++hist(min_idx);
        }
        //hist.print("\nhist");

        return hist;
    }

private:
    struct is_spmat
    {
        using type = typename std::is_same<Hist,
        arma::SpMat<typename Hist::elem_type>>::type;

        enum{value = std::is_same<type, std::true_type>::value};
    };

    struct is_mat
    {
        using type = typename std::is_same<Hist,
        arma::Mat<typename Hist::elem_type>>::type;

        enum{value = std::is_same<type, std::true_type>::value};
    };

    struct is_two_dim
    {
        using type  = typename std::conditional<is_spmat::value ||
        is_mat::value, std::true_type, std::false_type>::type;

        enum{value = std::is_same<type, std::true_type>::value};
    };

    struct is_col
    {
        using type = typename std::is_same<Hist,
        arma::Col<typename Hist::elem_type>>::type;

        enum{value = std::is_same<type, std::true_type>::value};
    };

    struct is_row
    {
        using type = typename std::is_same<Hist,
        arma::Row<typename Hist::elem_type>>::type;

        enum{value = std::is_same<type, std::true_type>::value};
    };

    struct is_arma
    {
        enum {value = is_spmat::value || is_mat::value ||
             is_row::value || is_col::value};
    };

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

    arma::Mat<T> code_book_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_BOVW_HPP
