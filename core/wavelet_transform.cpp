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

inline
float soft_shrink(float d,float T)
{
    auto const AbsD = std::abs(d);

    return AbsD > T ? sgn(d) * (AbsD - T) : 0;
}

inline
float hard_shrink(float d,float T)
{
    return std::abs(d) > T ? d : 0;
}

inline
float garrot_shrink(float d, float T)
{
    return std::abs(d) > T ? d - ((T * T) / d) : 0;
}

bool realloc_if_same(cv::Mat &src, cv::Mat &dst, int type)
{
    cv::Mat temp_dst;
    if(src.data == dst.data){
        temp_dst.create(src.rows, src.cols, type);
        dst = temp_dst;
        return true;
    }else{
        dst.create(src.rows, src.cols, type);
    }

    return false;
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
    bool IsSameAddr = realloc_if_same(src, dst, CV_32F);
    src.convertTo(src, CV_32F);
    int const Width = src.cols;
    int const Height = src.rows;
    for (int k = 0; k < n_iter; ++k){
        for (int y = 0; y < (Height>>(k+1)); ++y){
            for (int x=0; x<(Width>>(k+1)); ++x){
                auto const c = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x) = c;

                auto const dh = (src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x+(Width>>(k+1))) = dh;

                auto const dv = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(Height>>(k+1)),x) = dv;

                auto const dd = (src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(Height>>(k+1)),x+(Width>>(k+1)))=dd;
            }
        }
        dst.copyTo(src);
    }
    if(IsSameAddr){
        src = dst;
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
    bool IsSameAddr = realloc_if_same(src, dst, CV_32F);
    src.convertTo(src, CV_32F);
    int const Width = src.cols;
    int const Height = src.rows;
    //--------------------------------
    // NIter - number of iterations
    //--------------------------------
    for (int k = n_iter; k > 0; --k)
    {
        for (int y = 0; y < (Height>>k); ++y)
        {
            for (int x= 0; x < (Width>>k); ++x)
            {
                float const c = src.at<float>(y,x);
                float dh = src.at<float>(y,x+(Width>>k));
                float dv = src.at<float>(y+(Height>>k),x);
                float dd = src.at<float>(y+(Height>>k),x+(Width>>k));

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
        cv::Mat C = src(cv::Rect(0,0,Width>>(k-1),Height>>(k-1)));
        cv::Mat D = dst(cv::Rect(0,0,Width>>(k-1),Height>>(k-1)));
        D.copyTo(C);
    }
    if(IsSameAddr){
        src = dst;
    }
}

/**
 * @brief get the regions separated by haar wavelet transform.\n
 * If the n_iter == 1, return value will contain(by order) LL,HL,LH,HH.\n
 * If the n_iter == 2, return value will contain(by order)\N
 * LL2,HL2,LH2,HH2,HL,LH,HH and so on
 * @param size width and height of the image want to do haar wavelet transform
 * @param n_iter number of iteration
 * @return the regions separated by haar wavelet transform
 */
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
