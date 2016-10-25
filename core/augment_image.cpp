#include "augment_image.hpp"

namespace ocv{

namespace{

/*template<typename T>
void set_range(std::pair<T, T> const &input, std::pair<T, T> &output)
{
    if(input.first < input.second){
        output = input;
    }else{
        output.first = input.second;
        output.second = input.first;
    }
}//*/

template<typename T>
std::uniform_real_distribution<T> generate_distribution_range(T value)
{
    if(value < 0){
        value *= 1;
    }
    return std::uniform_real_distribution<T>(-value, value);
}

void translate_image(cv::Mat const &input, cv::Mat &output,
                     float horizontal_offset, float vertical_offset)
{
    cv::Mat const trans_mat = (cv::Mat_<float>(2,3) <<
                               1, 0, horizontal_offset * input.cols, 0, 1,
                               vertical_offset * input.rows);
    cv::warpAffine(input, output, trans_mat, input.size(),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT101);
}

}

void rotate_image(const cv::Mat &input, cv::Mat &output, double rotate, int border_type)
{
    cv::Point2f const pt(input.cols/2.0f, input.rows/2.0f);
    cv::Mat const rotation_mat = cv::getRotationMatrix2D(pt, rotate, 1.0);
    cv::warpAffine(input, output, rotation_mat, input.size(),
                   cv::INTER_LINEAR, border_type);
}

bool augment_image::get_flip_horizontal() const noexcept
{
    return flip_horizontal_;
}

bool augment_image::get_flip_vertical() const noexcept
{
    return flip_vertical_;
}

float augment_image::get_vertical_shift_range() const noexcept
{
    return vertical_shift_range_;
}

void augment_image::set_flip_horizontal(bool value) noexcept
{
    flip_horizontal_ = value;
}

void augment_image::set_flip_vertical(bool value) noexcept
{
    flip_vertical_ = value;
}

void augment_image::set_horizontal_shift_range(float value) noexcept
{
    horizontal_shift_range_ = value;
    shift_hor_distribution_ = generate_distribution_range(value);
}

void augment_image::set_random(bool value) noexcept
{
    random_ = value;
}

void augment_image::set_rotate_range(double value) noexcept
{
    rotate_angle_ = value;
    rotate_distribution_ = generate_distribution_range(value);
}

void augment_image::set_shift_horizontal(bool value) noexcept
{
    shift_horizontal_ = value;
}

void augment_image::set_shift_vertical(bool value) noexcept
{
    shift_vertical_ = value;
}

void augment_image::set_vertical_shift_range(float value) noexcept
{
    vertical_shift_range_ = value;
    shift_ver_distribution_ = generate_distribution_range(value);
}

void augment_image::flip(const cv::Mat &input, std::vector<cv::Mat> &results) const
{
    if(flip_horizontal_){
        cv::Mat temp;
        cv::flip(input, temp, 1);
        results.emplace_back(temp);
    }

    if(flip_vertical_){
        cv::Mat temp;
        cv::flip(input, temp, -1);
        results.emplace_back(temp);
    }
}

void augment_image::rotate(const cv::Mat &input, std::vector<cv::Mat> &results) const
{
    cv::Mat temp;
    if(random_){
        rotate_image(input, temp, rotate_distribution_(gen_), border_type_);
    }else{
        rotate_image(input, temp, rotate_angle_, border_type_);
    }
    results.emplace_back(temp);
}

std::vector<cv::Mat> augment_image::transform(const cv::Mat &input)
{
    std::vector<cv::Mat> results;

    flip(input, results);
    shift(input, results);
    rotate(input, results);

    return results;
}

std::vector<cv::Mat> augment_image::transform(const std::vector<cv::Mat> &input)
{
    std::vector<cv::Mat> results;
    for(auto const &img : input){
        auto output = transform(img);
        results.insert(std::end(results), std::begin(output), std::end(output));
    }

    return results;
}

void augment_image::shift(const cv::Mat &input, std::vector<cv::Mat> &results) const
{
    cv::Mat temp;
    if(random_){        
        translate_image(input, temp, shift_hor_distribution_(gen_), 0.f);
    }else{
        translate_image(input, temp, horizontal_shift_range_, 0.f);
    }
    results.emplace_back(temp.clone());

    if(random_){
        translate_image(input, temp, 0.f, shift_ver_distribution_(gen_));
    }else{
        translate_image(input, temp, 0.f, vertical_shift_range_);
    }
    results.emplace_back(temp);
}

bool augment_image::get_random() const noexcept
{
    return random_;
}

double augment_image::get_rotate_range() const noexcept
{
    return rotate_angle_;
}

bool augment_image::get_shift_vertical() const noexcept
{
    return shift_vertical_;
}

bool augment_image::get_shift_horizontal() const noexcept
{
    return shift_horizontal_;
}

float augment_image::get_horizontal_shift_range() const noexcept
{
    return horizontal_shift_range_;
}

}
