#include "wavelet_transform.hpp"

#include <cmath>

namespace ocv{

namespace{

float sgn(float x)
{
    float res = 0;
    if(x == 0){
        res = 0;
    }
    if(x > 0){
        res = 1;
    }
    if(x < 0){
        res = -1;
    }
    return res;
}

float soft_shrink(float d,float T)
{
    float res = 0;
    auto const AbsD = std::abs(d);
    if(AbsD > T){
        res = sgn(d) * (AbsD - T);
    }else{
        res = 0;
    }

    return res;
}

float hard_shrink(float d,float T)
{
    float res = 0;
    if(std::abs(d) > T){
        res = d;
    }else{
        res = 0;
    }

    return res;
}

float garrot_shrink(float d, float T)
{
    float res = 0;
    if(std::abs(d) > T){
        res = d - ((T * T) / d);
    }else{
        res = 0;
    }

    return res;
}

}

/**
 * @brief do the haar wavelet transform
 * @param src src of the image need to transform, this image\n
 * will be altered, beware
 * @param dst the output of the image after haar wavelet transform
 * @param n_iter number of iteration(level)
 */
void haar_wavelet(cv::Mat &src, cv::Mat &dst, int n_iter)
{
    cv::Mat temp_dst;
    if(src.data == dst.data){
        temp_dst.create(src.rows, src.cols, CV_32F);
        dst = temp_dst;
    }else{
        dst.create(src.rows, src.cols, CV_32F);
    }
    src.convertTo(src, CV_32F);
    int width = src.cols;
    int height = src.rows;
    for (int k = 0; k < n_iter; ++k){
        for (int y = 0; y < (height>>(k+1)); ++y){
            for (int x=0; x<(width>>(k+1)); ++x){
                auto const c = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x) = c;

                auto const dh = (src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x+(width>>(k+1))) = dh;

                auto const dv = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(height>>(k+1)),x) = dv;

                auto const dd = (src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(height>>(k+1)),x+(width>>(k+1)))=dd;
            }
        }
        dst.copyTo(src);
    }
}

/**
 * @brief do the inverse haar wavelet transform
 * @param src src of the image need to restore, this image\n
 * will be altered, beware
 * @param dst the output of the image after haar wavelet transform
 * @param n_iter number of iteration(level)
 */
void haar_wavelet_inv(cv::Mat &src, cv::Mat &dst,
                      int n_iter, HaarShrink shrinkage_type,
                      float shrinkage_t)
{
    cv::Mat temp_dst;
    if(src.data == dst.data){
        temp_dst.create(src.rows, src.cols, CV_32F);
        dst = temp_dst;
    }else{
        dst.create(src.rows, src.cols, CV_32F);
    }
    src.convertTo(src, CV_32F);
    int width = src.cols;
    int height = src.rows;
    //--------------------------------
    // NIter - number of iterations
    //--------------------------------
    for (int k = n_iter; k > 0; --k)
    {
        for (int y = 0; y < (height>>k); ++y)
        {
            for (int x= 0; x < (width>>k); ++x)
            {
                float c = src.at<float>(y,x);
                float dh = src.at<float>(y,x+(width>>k));
                float dv = src.at<float>(y+(height>>k),x);
                float dd = src.at<float>(y+(height>>k),x+(width>>k));

                // (shrinkage)
                switch(shrinkage_type)
                {
                case HaarShrink::HARD:
                    dh = hard_shrink(dh,shrinkage_t);
                    dv = hard_shrink(dv,shrinkage_t);
                    dd = hard_shrink(dd,shrinkage_t);
                    break;
                case HaarShrink::SOFT:
                    dh = soft_shrink(dh,shrinkage_t);
                    dv = soft_shrink(dv,shrinkage_t);
                    dd = soft_shrink(dd,shrinkage_t);
                    break;
                case HaarShrink::GARROT:
                    dh = garrot_shrink(dh,shrinkage_t);
                    dv = garrot_shrink(dv,shrinkage_t);
                    dd = garrot_shrink(dd,shrinkage_t);
                    break;
                }

                //-------------------
                dst.at<float>(y*2,x*2)=0.5f*(c+dh+dv+dd);
                dst.at<float>(y*2,x*2+1)=0.5f*(c-dh+dv-dd);
                dst.at<float>(y*2+1,x*2)=0.5f*(c+dh-dv-dd);
                dst.at<float>(y*2+1,x*2+1)=0.5f*(c-dh-dv+dd);
            }
        }
        cv::Mat C = src(cv::Rect(0,0,width>>(k-1),height>>(k-1)));
        cv::Mat D = dst(cv::Rect(0,0,width>>(k-1),height>>(k-1)));
        D.copyTo(C);
    }
}

std::vector<cv::Rect> haar_wavelet_region(cv::Size2i const &size,
                                          int n_iter)
{
    if(n_iter >= 1 && size.area() > 1){
        std::vector<cv::Rect> result;
        auto const Func = [&result](int width, int height)
        {
            result.emplace_back(width, 0, width, height);
            result.emplace_back(0, height, width, height);
            result.emplace_back(width, height, width, height);
        };
        if(n_iter == 1){
            auto const Width = size.width / 2;
            auto const Height = size.height / 2;
            if(Width > 0 && Height > 0){
                result.emplace_back(0, 0, Width, Height);
                Func(Width, Height);
            }
        }else{
            auto const Width = size.width >> n_iter;
            auto const Height = size.height >> n_iter;
            if(Width > 0 && Height > 0){
                result.emplace_back(0, 0, Width, Height);
                Func(Width, Height);
                for(int i = n_iter - 1; i > 0; --i){
                    Func(size.width >> i, size.height >> i);
                }
            }
        }

        return result;
    }

    return {};
}

}
