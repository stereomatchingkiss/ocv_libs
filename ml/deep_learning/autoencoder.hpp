#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <opencv2/core.hpp>

/*! \file autoencoder.hpp
    \brief implement the algorithm--autoencoder based on\n
    the description of UFLDL
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup ml
 *  @{
 */
namespace ml{

class autoencoder
{
public:
    struct layer_struct
    {
        layer_struct(int input_size, int hidden_size,
                     int mat_type,
                     double cost = 0);

        cv::Mat w1_;
        cv::Mat w2_;
        cv::Mat b1_;
        cv::Mat b2_;
        cv::Mat w1_grad_;
        cv::Mat w2_grad_;
        cv::Mat b1_grad_;
        cv::Mat b2_grad_;
        double cost_;
    };

    explicit autoencoder(cv::AutoBuffer<int> const &hidden_size);

    autoencoder& operator=(autoencoder const&) = delete;
    autoencoder& operator=(autoencoder &&) = delete;
    autoencoder(autoencoder const&) = delete;
    autoencoder(autoencoder &&) = delete;

    std::vector<layer_struct> const& get_layer_struct() const;

    void set_beta(double beta);
    void set_eps(double eps);
    void set_hidden_layer_size(cv::AutoBuffer<int> const &size);
    void set_lambda(double lambda);
    void set_learning_rate(double lrate);
    void set_max_iter(int iter);
    void set_sparse(double sparse);

    void train(cv::Mat const &input);

private:
    struct activation
    {
        void clear();

        cv::Mat hidden_;
        cv::Mat output_;
    };

    struct buffer
    {
        void clear();

        cv::Mat delta2_;
        cv::Mat delta3_;
        cv::Mat delta_buffer_;
        cv::Mat pj_; //the average activation of hidden units
    };

    struct params
    {
        params();

        double beta_;
        double eps_;
        cv::AutoBuffer<int> hidden_size_;
        double lambda_;
        double lrate_; //learning rate
        int max_iter_;
        double sparse_;
    };

    void encoder_cost(cv::Mat const &input,
                      layer_struct &es);
    void encoder_gradient(cv::Mat const &input,
                          layer_struct &es);

    void get_delta_2(cv::Mat const &delta_3,
                     layer_struct const &es);
    void get_activation(cv::Mat const &input,
                        layer_struct const &es);

    void update_weight_and_bias(cv::Mat const &bias,
                                cv::Mat &weight);

    activation act_;    
    buffer buffer_;
    std::vector<layer_struct> layers_;
    int mat_type_;
    params params_;
    double const zero_firewall_; //prevent divide by zero
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // AUTOENCODER_H
