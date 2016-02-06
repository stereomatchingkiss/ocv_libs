#ifndef OCV_IMAGE_CONVERTER_HPP
#define OCV_IMAGE_CONVERTER_HPP

#include <ocv_libs/core/for_each.hpp>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup tiny_cnn
 *  @{
 */
namespace tiny_cnn{

template<typename Img>
Img cvmat_to_img(cv::Mat const& input,
                 float_t scale_min = -1,
                 float_t scale_max = 1)
{
    CV_Assert(input.type() == CV_8U || input.type() == CV_8UC3);
    CV_Assert(scale_min < scale_max);

    Img img;
    if(input.type() == CV_8UC3){
        int const area = input.rows * input.cols;
        img.resize(area*3);
        for (int i = 0; i < input.rows; i++) { // Go over all rows
            int const width = input.cols*i;
            for (int j = 0; j < input.cols; j++) { // Go over all columns
                for (int c = 0; c < 3; c++) { // Go through all channels
                    img[area*c + width + j] =
                            input.at<cv::Vec3b>(i, j)[c]/255.0*
                            (scale_max - scale_min) + scale_min;
                }
            }
        }
    }else{
        int const area = input.rows * input.cols;
        img.resize(area);
        size_t index = 0;
        ocv::for_each_channels<uchar>(input, [&](uchar c)
        {
            img[index++] = scale_min + (scale_max - scale_min) * c / 255.0;
        });
    }

    return img;
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CVMAT_TO_TINY_VEC_HPP
