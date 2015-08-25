#ifndef CV_TO_EIGEN_HPP
#define CV_TO_EIGEN_HPP

#include <opencv2/core.hpp>

#include <Eigen/Dense>

/*! \file cv_to_eigen.hpp
    \brief collect some tools to transform cv::Mat to\n
    Eigen matrix
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup eigen
 *  @{
 */
namespace eigen{

template<typename T>
using MatRowMajor =
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
Eigen::RowMajor>;

template<typename T>
void cv2eigen_cpy(cv::Mat const &input, MatRowMajor<T> &output){
    output.resize(input.rows, input.cols);
    cv::Mat dst(input.rows, input.cols, cv::DataType<T>::type,
                static_cast<void*>(output.data()),
                static_cast<size_t>(output.stride()*sizeof(T)));
    input.convertTo(dst, dst.type());
}

template<typename T>
void eigen2cv_cpy(MatRowMajor<T> const &input, cv::Mat &output)
{
    cv::Mat src(input.rows(), input.cols(), cv::DataType<T>::type,
                (void*)input.data(),
                static_cast<size_t>(input.stride()*sizeof(T)));
    src.copyTo(output);
}

/**
 *@brief create memory align cv::Mat
 */
template<typename T>
class align_cv_mat
{
    static_assert(std::is_arithmetic<T>::value,
                  "T should be arithmetic type");
public:
    align_cv_mat() : buffer_{nullptr}{}
    ~align_cv_mat()
    {
        destroy();
    }

    align_cv_mat(align_cv_mat const&) = delete;
    align_cv_mat(align_cv_mat &&) = delete;
    align_cv_mat& operator=(align_cv_mat const&) = delete;
    align_cv_mat& operator=(align_cv_mat &&) = delete;

    /**
     * @brief create the memory align cv::Mat which meet\n
     * the memory alignment requirement of Eigen
     * @param rows the rows of the matrix
     * @param cols the cols of the matrix
     * @return memory align cv::Mat
     * @warning 1 : Do not release the memory of cv::Mat\n
     * by yourself, leave the resource management to align_mat.\n
     * This cv::Mat only borrow the buffer allocated by the\n
     * Eigen::aligned_allocator<T>.\n
     * 2 : Every time you create a new Mat, the older buffer\n
     * of the Mat will be released
     */
    cv::Mat create(int rows, int cols)
    {
        destroy();
        buffer_ = alloc_.allocate(rows*cols);
        return cv::Mat(rows, cols,
                       cv::DataType<T>::type,
                       static_cast<void*>(buffer_),
                       static_cast<size_t>(cols * sizeof(T)));
    }

    /**
     * @brief destroy the buffer
     */
    void destroy()
    {
        if(buffer_ != nullptr){
            alloc_.deallocate(reinterpret_cast<T*>(buffer_),
                              0);
            buffer_ = nullptr;
        }
    }

private:
    Eigen::aligned_allocator<T> alloc_;
    T *buffer_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // CV_TO_EIGEN_HPP

