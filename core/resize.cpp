#include "resize.hpp"

namespace ocv{

void resize_aspect_ratio(const cv::Mat &input, cv::Mat &output,
                         const cv::Size &size, int interpolation)
{
    CV_Assert(size.width != 0 || size.height != 0);

    if(size.width != 0){
        int const Height = input.rows * size.width / input.cols;
        cv::resize(input, output, {size.width, Height}, interpolation);
    }else{
        int const Width = input.cols * size.height / input.rows;
        cv::resize(input, output, {Width, size.height}, interpolation);
    }
}

}
