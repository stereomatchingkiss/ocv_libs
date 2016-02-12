#include "f2d_descriptor.hpp"

namespace ocv{

namespace cbir{

f2d_detector::f2d_detector(cv::Ptr<cv::FeatureDetector> detector,
                           cv::Ptr<cv::DescriptorExtractor> extractor)
    : detector_(detector),
      extractor_(extractor)
{

}

void f2d_detector::get_descriptor(const cv::Mat &input,
                                  cv::Mat &output)
{
    std::vector<cv::KeyPoint> keypoints;
    if(detector_ == extractor_){
        detector_->detectAndCompute(input, cv::noArray(),
                                    keypoints, output);
    }else{
        detector_->detect(input, keypoints);
        extractor_->compute(input, keypoints, output);
    }
}

cv::Mat f2d_detector::get_descriptor(const cv::Mat &input)
{
    cv::Mat output;
    get_descriptor(input, output);

    return output;
}

}

}
