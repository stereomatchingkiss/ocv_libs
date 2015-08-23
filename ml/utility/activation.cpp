#include "activation.hpp"

namespace ocv{

namespace ml{

void sigmoid::operator()(const cv::Mat &input, cv::Mat &output) const
{
    cv::multiply(input, -1.0, output);
    cv::exp(output, output);
    output += 1.0;
    cv::divide(1.0, output, output);//*/
}



}}
