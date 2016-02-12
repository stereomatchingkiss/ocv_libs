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
                                  result_type &output)
{
    if(detector_ == extractor_){
        detector_->detectAndCompute(input, cv::noArray(),
                                    output.first,
                                    output.second);
    }else{
        detector_->detect(input, output.first);
        extractor_->compute(input, output.first,
                            output.second);
    }
}

f2d_detector::result_type
f2d_detector::get_descriptor(const cv::Mat &input)
{
    result_type output;
    get_descriptor(input, output);

    return output;
}

}

}
