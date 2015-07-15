#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

namespace ocv{

enum class HaarShrink{
    HARD,
    SOFT,
    GARROT
};

void haar_wavelet(cv::Mat &src, cv::Mat &dst, int n_iter);

void haar_wavelet_inv(cv::Mat &src, cv::Mat &dst,
                      int n_iter,
                      HaarShrink shrinkage_type = HaarShrink::GARROT,
                      float shrinkage_t = 50);
}

#endif // WAVELET_TRANSFORM_HPP

