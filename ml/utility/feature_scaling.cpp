#include "feature_scaling.hpp"

namespace ocv{

namespace ml{

void z_score_scaling(cv::Mat const &input, cv::Mat &output)
{
    CV_Assert(input.channels() == 1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(input, mean, stddev);

    output -= mean.val[0];
    stddev.val[0] = stddev.val[0] != 0 ?
                stddev.val[0] : 0.0001;
    output /= stddev.val[0];
}


}}
