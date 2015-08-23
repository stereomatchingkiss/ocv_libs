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

void zero_mean(const cv::Mat &input, cv::Mat &output,
               sample_type type)
{
    output.create(input.size(), input.type());
    if(type == sample_type::col){
        for(int i = 0; i != input.cols; ++i){
            cv::Scalar mean = 0;
            cv::meanStdDev(input.col(i), mean, cv::Scalar());
            cv::subtract(input.col(i), mean.val[0],
                    output.col(i));
        }
    }else{
        for(int i = 0; i != input.rows; ++i){
            cv::Scalar mean = 0;
            cv::meanStdDev(input.row(i), mean, cv::Scalar());
            cv::subtract(input.row(i), mean.val[0],
                    output.row(i));
        }
    }
}


}}
