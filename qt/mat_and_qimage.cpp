#include <opencv2/imgproc/imgproc.hpp>

#include "mat_and_qimage.hpp"

namespace ocv{ namespace qt{

namespace
{

/**
 * @brief copy QImage into cv::Mat
 */
struct mat_to_qimage_cpy_policy
{
    static QImage start(cv::Mat const &mat, QImage::Format format)
    {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, format).copy();
    }
};

struct mat_to_qimage_ref_policy
{
    static QImage start(cv::Mat &mat, QImage::Format format)
    {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, format);
    }
};

/**
 * @brief generic class for reducing duplicate codes
 */
template<typename Policy>
struct mat_to_qimage
{
    template<typename Image>
    static QImage run(Image &&mat, bool swap);
};

template<typename Policy>
template<typename Image>
QImage mat_to_qimage<Policy>::run(Image &&mat, bool swap)
{
    if(!mat.empty()){
        switch(mat.type()){

        case CV_8UC3 :{
            if(swap){
                return Policy::start(mat, QImage::Format_RGB888).rgbSwapped();
            }else{
                return Policy::start(mat, QImage::Format_RGB888);
            }
        }

        case CV_8U :{
            return Policy::start(mat, QImage::Format_Indexed8);
        }

        case CV_8UC4 :{
           return Policy::start(mat, QImage::Format_ARGB32);
        }

        }
    }

    return {};
}

/**
 * @brief copy QImage into cv::Mat
 */
struct qimage_to_mat_cpy_policy
{
    static cv::Mat start(QImage const &img, int format)
    {
        return cv::Mat(img.height(), img.width(), format, const_cast<uchar*>(img.bits()), img.bytesPerLine()).clone();
    }
};

/**
 * @brief make Qimage and cv::Mat share the same buffer, the resource
 * of the cv::Mat must not deleted before the QImage finish
 * the jobs.
 */
struct qimage_to_mat_ref_policy
{
    static cv::Mat start(QImage &img, int format)
    {
        return cv::Mat(img.height(), img.width(), format, img.bits(), img.bytesPerLine());
    }
};

/**
 * @brief generic class for reducing duplicate codes
 */
template<typename Policy>
struct qimage_to_mat
{
    template<typename Image>
    static cv::Mat run(Image &&img, bool swap);
};

/**
 *@brief transform QImage to cv::Mat
 *@param img : input image
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
template<typename Policy>
template<typename Image>
cv::Mat qimage_to_mat<Policy>::run(Image &&img, bool swap)
{
    if(img.isNull()){
        return cv::Mat();
    }

    switch (img.format()) {
    case QImage::Format_RGB888:{
        auto result = Policy::start(img, CV_8UC3);
        if(swap){
            cv::cvtColor(result, result, CV_RGB2BGR);
        }
        return result;
    }
    case QImage::Format_Indexed8:{
        return Policy::start(img, CV_8U);
    }
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:{
        return Policy::start(img, CV_8UC4);
    }
    default:
        break;
    }

    return {};
}

} //end of namespace

/**
 *@brief copy cv::Mat into QImage
 *
 *@param mat : input mat
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
QImage mat_to_qimage_cpy(cv::Mat const &mat, bool swap)
{
    //return {};
    return mat_to_qimage<mat_to_qimage_cpy_policy>::run(mat, swap);
}

/**
 *@brief make Qimage and cv::Mat share the same buffer, the resource
 * of the cv::Mat must not deleted before the QImage finish
 * the jobs.
 *
 *@param mat : input mat
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
QImage mat_to_qimage_ref(cv::Mat &mat, bool swap)
{
    return mat_to_qimage<mat_to_qimage_ref_policy>::run(mat, swap);
}

/**
 *@brief transform QImage to cv::Mat by copy QImage to cv::Mat
 *@param img : input image
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
cv::Mat qimage_to_mat_cpy(QImage const &img, bool swap)
{
    return qimage_to_mat<qimage_to_mat_cpy_policy>::run(img, swap);
}

/**
 *@brief transform QImage to cv::Mat by sharing the buffer
 *@param img : input image
 *@param swap : true : swap RGB to BGR; false, do nothing
 */
cv::Mat qimage_to_mat_ref(QImage &img, bool swap)
{
    return qimage_to_mat<qimage_to_mat_ref_policy>::run(img, swap);
}

}}
