#ifndef OCV_GENERICFOREACH_HPP
#define OCV_GENERICFOREACH_HPP

#include <opencv2/core.hpp>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * merge overlap rectangles, thanks for stackoverflow
 * http://stackoverflow.com/questions/29523177/opencv-merging-overlapping-rectangles
 */
class merge_overlap_rectangle
{
public:
    /**
     * merge overlap rectange
     * @param mask_size size of the image
     * @param input rectangles within the image
     * @param output merged rectangles
     */
    void merge(cv::Size const &image_size,
               std::vector<cv::Rect> const &input,
               std::vector<cv::Rect> &output);
private:
    std::vector<std::vector<cv::Point>> contours_;
    cv::Mat mask_;
};

} /*! @} End of Doxygen Groups*/

#endif
