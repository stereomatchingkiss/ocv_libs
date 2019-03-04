#ifndef OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_FILTER_HPP
#define OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_FILTER_HPP

#include "coco_item_type.hpp"

#include <opencv2/core.hpp>

#include <mxnet-cpp/MxNetCpp.h>

#include <deque>
#include <vector>

namespace ocv{

namespace obj_det{

class mxnet_object_detector_filter
{
public:
    struct result_type
    {
        float confidence_ = 0.0f;
        coco_item_type item_;
        cv::Rect roi_;
    };

    explicit mxnet_object_detector_filter(std::vector<coco_item_type> const &items_to_detect,
                                          cv::Size const &obj_detector_process_size,
                                          float min_confidence = 0.4f);

    std::vector<result_type> filter(std::vector<mxnet::cpp::NDArray> const &input, cv::Size const &image_size) const;

    void set_min_confidence(float input) noexcept;
    void set_items_to_detect(std::vector<coco_item_type> const &items_to_detect);
    void set_obj_detector_process_size(cv::Size const &obj_detector_process_size) noexcept;

private:
    cv::Rect clip_points(float x1, float y1, float x2, float y2, cv::Size const &image_size) const noexcept;
    cv::Rect normalize_points(float x1, float y1, float x2, float y2, cv::Size const &image_size) const noexcept;

    float min_confidence_;
    std::deque<bool> items_to_detect_;
    cv::Size obj_detector_process_size_;
};

}

}

#endif // OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_FILTER_HPP
