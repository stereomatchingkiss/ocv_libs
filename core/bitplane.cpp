#include "bitplane.hpp"

#include "for_each.hpp"

#include <tbb/tbb.h>

void bitplane_generator(const cv::Mat &input, std::vector<cv::Mat> &output)
{
    CV_Assert(input.type() == CV_8U);

    output.resize(8);
    uchar const mask[] = {1, 2, 4, 8, 16, 32, 64, 128};

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 8),
                      [&](tbb::blocked_range<size_t> const &r)
    {
        for(size_t i = r.begin(); i != r.end(); ++i){
            ocv::for_each_channels<uchar>(input, output[i],
                                          [&](uchar a, uchar b)
            {
                b = a & mask[i];
            });
        }
    });

    /*for(size_t i = 0; i != output.size(); ++i){
        ocv::for_each<uchar>(input, output[i],
                             [=](uchar a, uchar &b)
        {
            std::bitset<sizeof(uchar)> bits(a);
            b = bits[i];
        });
    }//*/
}
