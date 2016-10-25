#ifndef OCV_CORE_AUGMENT_IMAGE_HPP
#define OCV_CORE_AUGMENT_IMAGE_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <random>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 * @brief rotate image on center
 * @param input self explain
 * @param output self explain
 * @param rotate self explain
 * @param border_type same as BorderTypes of opencv
 */
void rotate_image(cv::Mat const &input, cv::Mat &output, double rotate,
                  int border_type = cv::BORDER_CONSTANT);

/**
 * @brief Perform several augmentation on the images at once
 */
class augment_image
{
public:
    augment_image() = default;

    augment_image(augment_image const&) = delete;
    augment_image& operator=(augment_image const&) = delete;
    augment_image(augment_image&&) = delete;
    augment_image& operator=(augment_image&&) = delete;

    int get_border_type() const noexcept;
    bool get_flip_horizontal() const noexcept;
    bool get_flip_vertical() const noexcept;
    float get_horizontal_shift_range() const noexcept;
    bool get_random() const noexcept;
    double get_rotate_range() const noexcept;
    bool get_shift_vertical() const noexcept;
    bool get_shift_horizontal() const noexcept;
    float get_vertical_shift_range() const noexcept;

    void set_border_type(int value) noexcept;
    void set_flip_horizontal(bool value) noexcept;
    void set_flip_vertical(bool value) noexcept;
    void set_horizontal_shift_range(float value) noexcept;
    /**
     * @brief By default, random is off, if you set random as true,
     * the range of shift and rotation will vary between [-range, range]
     * @param value
     */
    void set_random(bool value) noexcept;
    void set_rotate_range(double value) noexcept;
    void set_shift_horizontal(bool value) noexcept;
    void set_shift_vertical(bool value) noexcept;
    void set_vertical_shift_range(float value) noexcept;

    std::vector<cv::Mat> transform(cv::Mat const &input);
    std::vector<cv::Mat> transform(std::vector<cv::Mat> const &input);

private:
    void flip(const cv::Mat &input, std::vector<cv::Mat> &results) const;
    void rotate(cv::Mat const &input, std::vector<cv::Mat> &results) const;
    void shift(cv::Mat const &input, std::vector<cv::Mat> &results) const;

    int border_type_ = cv::BORDER_CONSTANT;
    bool flip_horizontal_ = true;
    bool flip_vertical_ = true;
    mutable std::mt19937 gen_;
    float horizontal_shift_range_ = 0.2f;
    bool random_ = true;
    bool rotate_ = true;
    double rotate_angle_ = 20.0;

    bool shift_horizontal_ = true;
    bool shift_vertical_ = true;
    float vertical_shift_range_ = 0.2f;

    std::uniform_real_distribution<double> rotate_distribution_
    {-rotate_angle_, rotate_angle_};
    std::uniform_real_distribution<float> shift_hor_distribution_
    {-horizontal_shift_range_, horizontal_shift_range_};
    std::uniform_real_distribution<float> shift_ver_distribution_
    {-vertical_shift_range_, vertical_shift_range_};
};

} /*! @} End of Doxygen Groups*/

#endif // AUGMENT_IMAGE_HPP
