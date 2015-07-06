#include "io_img.hpp"

#include "mat_and_qimage.hpp"

#include <opencv2/highgui.hpp>

namespace ocv{ namespace qt{

cv::Mat read_cv_mat(const QString &file)
{
    cv::Mat img = cv::imread(file.toStdString());
    if(img.empty()){
        QImage qimg(file);
        if(!qimg.isNull()){
            img = qimage_to_mat_cpy(qimg);
        }
    }

    return img;
}

}}
