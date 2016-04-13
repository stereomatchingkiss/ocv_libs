#ifndef OCV_CBIR_FEATURES_INDEXER_HPP
#define OCV_CBIR_FEATURES_INDEXER_HPP

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

#include <functional>
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
 * ---------key points-----------------
 * 0 : cv::keypoint
 * 1 : cv::keypoint
 * ............................
 *
 * index 0 specify the range of the features of
 * image 0(ukbench03000.jpg) are located from
 * [0 to 594)
 *
 * index 1 specify the range of the features of
 * image 1(ukbench03001.jpg) are located from
 * [594 to 1027)
 * @code
 * features_indexer fi("ukbench.h5");
 * //largest size of the file, feature size, type of the feature
 * fi.create_dataset(16, 61, CV_8U);
 * fi.set_buffer_size(1000);
 * cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
 * cv::Ptr<cv::AKAZE> descriptor = detector;
 * ocv::cbir::f2d_detector f2d(detector, descriptor);
 * for(auto &pair : files){
 *   //pair : first is cv::Mat, second is image name
 *   //result : first is keypoints, second is feature
 *   auto result = f2d.get_descriptor(pair.first);
 *   fi.add_features(pair.second, result.first,
 *                   result.second);
 * }
 * fi.flush();
 * @endcode
 */
class features_indexer
{
public:
    /**
     * Overload of constructor, allow to open/create
     * the hdf5 later on
     */
    features_indexer();

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
     * @param keypoints keypoints of the features
     * @param features features of the image     
     * @warning This function require the cols of every features
     * remain the same.
     */
    void add_features(std::string const &image_name,
                      std::vector<cv::KeyPoint> const &keypoints,
                      cv::Mat const &features);

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
    std::vector<int> get_key_dimension() const;
    std::vector<int> get_names_dimension() const;

    void open(std::string const &file_name);

    /**
     * Read features
     * @param features self explained
     * @param begin begin index
     * @param end end index, must > begin
     */
    void read_features(cv::InputOutputArray &features,
                       int begin,
                       int end) const;

    /**
     * Read the features index associate with specific image
     * @param features_index feauteres index of specific image
     * @param image_index index of the image
     */
    void read_features_index(cv::InputOutputArray &features_index,
                             int image_index) const;

    /**
     * Read the keypoints associate with specific image
     * @param keypoints keypoints of the image
     * @param image_index index of the image
     */
    void read_keypoints(std::vector<cv::KeyPoint> &keypoints,
                        int image_index) const;

    /**
     * Read all of the image names
     * @param img_names store the name of the images
     */
    void read_image_name(std::vector<std::string> &img_names) const;

    /**
     * Read the name of the images
     * @param img_names name of the images
     * @param begin begin index
     * @param end end index, must > begin
     */
    void read_image_name(std::vector<std::string> &img_names,
                         int begin, int end) const;

    /**
     * Read features associated with specific image
     * @param features features of the image
     * @param image_index index of the image
     */
    void read_image_features(cv::InputOutputArray &features,
                             int image_index) const;    

    /**
     * Read features randomly
     * @param ratio how many percent want to read
     * @param read_func provide the way to read the features,
     * first parameter is the Mat with the features, second parameter
     * is the index of the features
     * @param seed random seed, easier for users to regenerate the results
     * @code
     * fi.read_random_features(0.25,
     * [&](cv::Mat const &features, int index)
     * {
     *     auto *fptr = features.ptr<uchar>(0);
     *     std::copy(fptr, fptr + features.cols, data.colptr(index));
     * });
     * @endcode
     */
    void read_random_features(double ratio,
                              std::function<void(cv::Mat const&, int)> read_func,
                              unsigned int seed = 0) const;

    /**
     * Read features randomly
     * @param feature_size how many features want to read
     * @param read_func provide the way to read the features,
     * first parameter is the Mat with the features, second parameter
     * is the index of the features
     * @param seed random seed, easier for users to regenerate the results
     * @code
     * fi.read_random_features(100000,
     * [&](cv::Mat const &features, int index)
     * {
     *     auto *fptr = features.ptr<uchar>(0);
     *     std::copy(fptr, fptr + features.cols, data.colptr(index));
     * });
     * @endcode
     */
    void read_random_features(int feature_size,
                              std::function<void(cv::Mat const&, int)> read_func,
                              unsigned int seed = 0) const;

    /**
     * Read the data of hdf5
     * @param features store features
     * @param features_index store featurs index     
     * @param image_names store image_name
     * @param keypoints keypoints of the features
     * @param img_begin the index of begin image
     * @param img_end the index of last image, img_end must >
     * img_begin
     */
    void read_data(cv::InputOutputArray &features,
                   cv::InputOutputArray &features_index,                                      
                   std::vector<cv::KeyPoint> &keypoints,
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
    std::string file_name_;
    cv::Ptr<cv::hdf::HDF5> h5io_;    
    std::string img_names_;    
    int image_row_offset_;
    std::vector<int> index_;
    std::vector<cv::KeyPoint> keypoints_;
    int name_size_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_FEATURES_INDEXER_HPP
