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

//--------------------------------
// Wavelet transform
//--------------------------------
void haar_wavelet(cv::Mat const &src, cv::Mat &dst, int n_iter)
{
    float c,dh,dv,dd;
    assert( src.type() == CV_32FC1 );
    assert( dst.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    for (int k = 0; k < n_iter; ++k)
    {
        for (int y=0;y<(height>>(k+1));y++)
        {
            for (int x=0; x<(width>>(k+1));x++)
            {
                c = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x) = c;

                dh = (src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y,x+(width>>(k+1))) = dh;

                dv = (src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(height>>(k+1)),x) = dv;

                dd = (src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5f;
                dst.at<float>(y+(height>>(k+1)),x+(width>>(k+1)))=dd;
            }
        }
        dst.copyTo(src);
    }
}

//--------------------------------
//Inverse wavelet transform
//--------------------------------
void haar_wavelet_inv(cv::Mat const &src, cv::Mat &dst,
                      int n_iter, Haar shrinkage_type,
                      float shrinkage_t)
{
    float c,dh,dv,dd;
    CV_Assert( src.type() == CV_32FC1 );
    CV_Assert( dst.type() == CV_32FC1 );
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
                c = src.at<float>(y,x);
                dh = src.at<float>(y,x+(width>>k));
                dv = src.at<float>(y+(height>>k),x);
                dd = src.at<float>(y+(height>>k),x+(width>>k));

                // (shrinkage)
                switch(shrinkage_type)
                {
                case Haar::HARD:
                    dh = hard_shrink(dh,shrinkage_t);
                    dv = hard_shrink(dv,shrinkage_t);
                    dd = hard_shrink(dd,shrinkage_t);
                    break;
                case Haar::SOFT:
                    dh = soft_shrink(dh,shrinkage_t);
                    dv = soft_shrink(dv,shrinkage_t);
                    dd = soft_shrink(dd,shrinkage_t);
                    break;
                case Haar::GARROT:
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

}
