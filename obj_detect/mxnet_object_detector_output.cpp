#include "mxnet_object_detector_output.hpp"

#include <mxnet-cpp/MxNetCpp.h>

#include <opencv2/imgproc.hpp>

#include <ocv_libs/mxnet/common.hpp>

using namespace mxnet::cpp;
using namespace ocv::mxnet_aux;

namespace ocv{

namespace obj_det{

mxnet_object_detector_output::mxnet_object_detector_output(std::string const &model_params,
                                                           std::string const &model_symbols,
                                                           Context const &context,
                                                           cv::Size const &process_size) :
    context_(new Context(context.GetDeviceType(), context.GetDeviceId())),
    process_size_(process_size)
{    
    executor_ = create_executor(model_params,
                                model_symbols,
                                context,
                                Shape(1, static_cast<unsigned>(process_size.height),
                                      static_cast<unsigned>(process_size.width), 3));
}

mxnet_object_detector_output::~mxnet_object_detector_output()
{

}

void mxnet_object_detector_output::forward(const cv::Mat &input)
{
    //By default, input_size_.height equal to 256 input_size_.width equal to 320.
    //Yolo v3 has a limitation, width and height of the image must be divided by 32.
    if(input.rows != process_size_.height || input.cols != process_size_.width){
        cv::resize(input, resize_img_, process_size_);
    }else{
        resize_img_ = input;
    }

    auto data = cvmat_to_ndarray(resize_img_, *context_);
    //Copy the data of the image to the "data"
    data.CopyTo(&executor_->arg_dict()["data"]);
    //Forward is an async api.
    executor_->Forward(false);
}

mxnet::cpp::NDArray mxnet_object_detector_output::get_bboxes() const
{
    return executor_->outputs[2].Copy(Context::cpu());
}

mxnet::cpp::NDArray mxnet_object_detector_output::get_labels() const
{
    return executor_->outputs[0].Copy(Context::cpu());
}

std::vector<NDArray> mxnet_object_detector_output::get_outputs() const
{
    return executor_->outputs;
}

mxnet::cpp::NDArray mxnet_object_detector_output::get_scores() const
{
    return executor_->outputs[1].Copy(Context::cpu());
}

}

}
