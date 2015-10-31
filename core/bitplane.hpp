#ifndef BITPLANE_HPP
#define BITPLANE_HPP

#include <opencv2/core.hpp>

#include <vector>

void bitplane_generator(cv::Mat const &input, std::vector<cv::Mat> &output);

#endif // BITPLANE_HPP

