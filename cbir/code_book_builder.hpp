#ifndef OCV_CBIR_CODE_BOOK_BUILDER_HPP
#define OCV_CBIR_CODE_BOOK_BUILDER_HPP

#include "features_indexer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

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
 *build the code book by kmeans, depend on
 *armadillo, opencv, hdf5
 *
 *@tparam T type of code book
 *@tparam U type of the features
 */
template<typename T>
class code_book_builder
{
public:
    /**
     * @param fi the feature indexer which store the
     * hdf5 data
     * @param sample_size size of the sample
     * @param feature_type type of the features
     * @param seed random seed
     */
    code_book_builder(features_indexer const &fi,
                      int sample_size,
                      int feature_type,
                      unsigned int seed = 0) :
        feature_dimension_(fi.get_features_dimension()),
        fi_(fi)
    {
        read_data(sample_size, feature_type, seed);
    }

    /**
     * @param fi the feature indexer which store the
     * hdf5 data
     * @param ratio ratio of the feature you want to sample
     * @param feature_type type of the features
     * @param seed random seed
     */
    code_book_builder(features_indexer const &fi,
                      double ratio,
                      int feature_type,
                      unsigned int seed = 0) :
        feature_dimension_(fi.get_features_dimension()),
        fi_(fi)
    {
        read_data(ratio, feature_type, seed);
    }

    /**
     * create code book
     * @param cluster cluster size, it should << than feature size
     * @param n_iter number of iteration, 10 is default number
     * @param print_mode true will print the else and vice versa
     */
    void create_code_book(arma::uword cluster,
                          arma::uword n_iter = 10,
                          bool print_mode = false)
    {
        arma::kmeans(code_book_, data_, cluster,
                     arma::random_subset, n_iter, print_mode);
    }

    /**
     * read features from hdf5
     * @param sample_size how many sample want to read
     * @param feature_type type of the features
     * @param seed random seed
     */
    void read_data(int sample_size, int feature_type,
                   unsigned int seed = 0)
    {
        data_.set_size(feature_dimension_[1], sample_size);
        switch(feature_type){
        case CV_8U :{
            fi_.read_random_features(sample_size,
                                     read_func<uchar>(data_), seed);
            break;
        }
        case CV_8S :{
            fi_.read_random_features(sample_size,
                                     read_func<schar>(data_), seed);
            break;
        }
        case CV_16U :{
            fi_.read_random_features(sample_size,
                                     read_func<ushort>(data_), seed);
            break;
        }
        case CV_16S :{
            fi_.read_random_features(sample_size,
                                     read_func<short>(data_), seed);
            break;
        }
        case CV_32S :{
            fi_.read_random_features(sample_size,
                                     read_func<int>(data_), seed);
            break;
        }
        case CV_32F :{
            fi_.read_random_features(sample_size,
                                     read_func<float>(data_), seed);
            break;
        }
        case CV_64F :{
            fi_.read_random_features(sample_size,
                                     read_func<double>(data_), seed);
            break;
        }
        default:{
            throw std::runtime_error("code_book_builder do not support this type");
        }
        }
    }

    /**
     * read features from hdf5
     * @param ratio ratio of the feature you want to sample
     * @param feature_type type of the features
     * @param seed random seed
     */
    void read_data(double ratio, int feature_type,
                   unsigned int seed = 0)
    {
        int const feature_size =
                static_cast<int>(feature_dimension_[0] * ratio);
        read_data(feature_size, feature_type, seed);
    }

    /**
     * get the code book
     * @return code book
     */
    arma::Mat<T> const& get_code_book() const
    {
        return code_book_;
    }

private:
    template<typename U>
    struct read_func
    {
        read_func(arma::Mat<T> &data) :
            data(data){}

        void operator()(cv::Mat const &features, int index)
        {
            auto *fptr = features.ptr<U>(0);
            //do not prefer std::copy since there are type cast warning
            //std::copy(fptr, fptr + features.cols, data.colptr(index));
            auto *cptr = data.colptr(index);
            for(int i = 0; i != features.cols; ++i){
                cptr[i] = fptr[i];
            }
        }

        arma::Mat<T> &data;
    };

    arma::Mat<T> code_book_;
    arma::Mat<T> data_;
    std::vector<int> feature_dimension_;
    features_indexer const &fi_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // CODE_BOOK_BUILDER_H
