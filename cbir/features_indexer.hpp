#ifndef OCV_CBIR_FEATURES_INDEXER_HPP
#define OCV_CBIR_FEATURES_INDEXER_HPP

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

#include <string>
#include <vector>

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
 * Save features of imgs into hdf5 format, the structures of the
 * hdf5 are
 *
 * ---------image_name----------------
 * 0 : ukbench03000.jpg
 * 1 : ukbench03001.jpg
 * ....................
 *
 * ---------index----------------------
 * 0 : 0, 594
 * 1 : 594, 1027
 * ....................
 *
 * ---------features-------------------
 * 0 : 32,33,44.....255
 * 1 : 45,88,99,.....,11
 * .....................
 *
 * index 0 specify the range of the features of
 * image 0(ukbench03000.jpg) are located from
 * [0 to 594)
 *
 * index 1 specify the range of the features of
 * image 1(ukbench03001.jpg) are located from
 * [594 to 1027)
 */
class features_indexer
{
public:
    /**
     * Construct hdf5 file with file_name
     * @param file_name name of the hdf5 file
     */
    explicit features_indexer(std::string const &file_name);

    /**
     * Destructor, will called flush()
     */
    ~features_indexer();

    /**
     * Add features into the data set.This function would not
     * write the features into the hdf5 instantly, it will cache
     * the features and flush them out when features buffer large
     * enough. You could call flush() to flush the data manually,
     * call set_buffer_size to adjust the size of the buffer.
     * Destructror will flush the features to hdf5
     * @param image_name name of the image
     * @param features features of the image
     * @warning This function require the cols of every features
     * remain the same.
     */
    void add_features(std::string const &image_name,
                      cv::Mat features);

    /**
     * Create dataset of the file
     * @param name_size Size of the image name(per row), if it is
     * unlimited(H5_UNLIMITED), when you read the name, it will
     * read as much names as possible
     * @param features_size Size of the features(per row), if it is
     * unlimited(H5_UNLIMITED), when you read the features, it will
     * read as much features as possible
     * @param features_type type of the features
     */
    void create_dataset(int name_size,
                        int features_size,
                        int features_type);

    /**
     * Write the data into hdf5
     */
    void flush();

    std::vector<int> get_features_dimension() const;
    std::vector<int> get_index_dimension() const;
    std::vector<int> get_names_dimension() const;

    /**
     * Read features associated with specific image
     * @param features features of the image
     * @param image_index index of the image
     */
    void read_features(cv::InputOutputArray &features,
                       int image_index) const;

    /**
     * Read the features index associate with specific image
     * @param features_index feauteres index of specific image
     * @param image_index index of the image
     */
    void read_features_index(cv::InputOutputArray &features_index,
                             int image_index) const;
    /**
     * Read the data of hdf5
     * @param features store features
     * @param features_index store featurs index
     * @param image_names store image_name
     * @param img_begin the index of begin image
     * @param img_end the index of last image, img_end must >=
     * img_begin
     */
    void read_data(cv::InputOutputArray &features,
                   cv::InputOutputArray &features_index,
                   std::vector<std::string> &image_names,
                   int img_begin, int img_end) const;

    /**
     * Set the size of the buffer, the size of the buffer
     * determine how many features will be cached before
     * the data write into to hdf5
     * @param value size of the buffer
     */
    void set_buffer_size(size_t value);

private:
    std::vector<int> get_dimension(std::string const &label) const;

    size_t buffer_size_;
    size_t cur_buffer_size_;
    cv::Mat features_;
    int feature_row_offset_;
    int features_size_;
    std::vector<int> index_;
    cv::Ptr<cv::hdf::HDF5> h5io_;
    std::string img_names_;
    int image_row_offset_;
    int name_size_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_FEATURES_INDEXER_HPP
