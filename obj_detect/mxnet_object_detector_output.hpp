#ifndef OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_OUTPUT_HPP
#define OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_OUTPUT_HPP

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace mxnet{

namespace cpp{

class Context;
class Executor;
class NDArray;

}

}

namespace ocv{

namespace obj_det{

class mxnet_object_detector_output
{
public:
    mxnet_object_detector_output(std::string const &model_params,
                                 std::string const &model_symbols,
                                 mxnet::cpp::Context const &context,
                                 cv::Size const &process_size = cv::Size(320, 256));
    ~mxnet_object_detector_output();

    mxnet_object_detector_output(mxnet_object_detector_output&&) = default;
    mxnet_object_detector_output& operator=(mxnet_object_detector_output&&) = default;

    void forward(cv::Mat const &input);

    mxnet::cpp::NDArray get_bboxes() const;
    mxnet::cpp::NDArray get_labels() const;
    std::vector<mxnet::cpp::NDArray> get_outputs() const;
    mxnet::cpp::NDArray get_scores() const;

private:
    std::unique_ptr<mxnet::cpp::Context> context_;
    std::unique_ptr<mxnet::cpp::Executor> executor_;
    cv::Size process_size_;
    cv::Mat resize_img_;
};

}

}

#endif // OCV_OBJ_DET_MXNET_OBJECT_DETECTOR_OUTPUT_HPP
