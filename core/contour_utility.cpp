#include "contour_utility.hpp"

namespace ocv{

void find_contours(cv::Mat const &img, std::vector<std::vector<cv::Point>> &contours,
                   cv::Range const &range,
                   int mode, int method)
{
    using contour = std::vector<cv::Point>;

    cv::findContours(img, contours, mode, method);
    auto it = std::remove_if(std::begin(contours), std::end(contours), [=](contour const &c)
    {
        auto const area = cv::boundingRect(c).area();
        return area < range.start || area > range.end;
    });
    contours.erase(it, std::end(contours));
}

std::vector<std::vector<cv::Point>> find_contours(cv::Mat const &img, cv::Range const &range,
                                                  int mode, int method)
{
    std::vector<std::vector<cv::Point>> contours;
    find_contours(img, contours, range, mode, method);

    return contours;
}

std::vector<std::vector<cv::Point>> find_contours(const cv::Mat &img, int mode, int method)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, mode, method);

    return contours;
}

void draw_convex(std::vector<std::vector<cv::Point>> const &contours, cv::Mat &img,
                 cv::Scalar const &color)
{
    std::vector<std::vector<cv::Point>> convex(1);
    for(size_t i = 0; i != contours.size(); ++i){
        cv::convexHull(contours[i], convex[0]);
        cv::drawContours(img, convex, -1, color);
    }
}

void draw_rect(const std::vector<std::vector<cv::Point>> &contours, cv::Mat &img,
               cv::Scalar const &color)
{
    for(size_t i = 0; i != contours.size(); ++i){
        auto const rect = cv::boundingRect(contours[i]);
        cv::rectangle(img, rect, color);
    }
}

void contours_to_bounding_rect(const std::vector<std::vector<cv::Point>> &contours,
                               std::vector<cv::Rect> &rects)
{
    rects.resize(contours.size());
    std::transform(std::begin(contours), std::end(contours),
                   std::begin(rects),
                   [](auto const &contour)
    {
        return cv::boundingRect(contour);
    });
}

std::vector<cv::Rect> contours_to_bounding_rect(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<cv::Rect> rects;
    contours_to_bounding_rect(contours, rects);

    return rects;
}

}
