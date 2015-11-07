#include "merge_rectangle.hpp"

#include <opencv2/imgproc.hpp>

namespace ocv{

void merge_overlap_rectangle::merge(cv::Size const &image_size,
                            std::vector<cv::Rect> const &input,
                            std::vector<cv::Rect> &output)
{
    cv::Size const ScaleFactor(10,10);
    //To expand rectangles,
    //i.e. increase sensitivity to nearby rectangles.
    //Doesn't have to be (10,10)--can be anything
    mask_.create(image_size, CV_8U);
    mask_ = 0;
    for(auto const &Rect : input)
    {
        cv::rectangle(mask_, Rect + ScaleFactor,
        {255}, CV_FILLED);
    }

    // Find contours in mask
    // If bounding boxes overlap, they will be joined by this function call
    contours_.clear();
    cv::findContours(mask_, contours_, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    output.clear();
    for(auto const &contour : contours_)
    {
        output.push_back(cv::boundingRect(contour));
    }
}

}
