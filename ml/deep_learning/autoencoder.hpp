#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <opencv2/core.hpp>

#include <random>

/*! \file autoencoder.hpp
    \brief implement the algorithm--autoencoder based on\n
    the description of UFLDL, these codes are develop based\n
    on the example on the website(http://eric-yuan.me/simple-deep-network/)
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
        layer_struct();
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

    cv::Mat const& get_activation() const;
    std::vector<layer_struct> const& get_layer_struct() const;

    void read(std::string const &file);

    void set_batch_fraction(int divide);
    void set_beta(double beta);
    void set_eps(double eps);
    void set_hidden_layer_size(cv::AutoBuffer<int> const &size);
    void set_lambda(double lambda);
    void set_learning_rate(double lrate);
    void set_max_iter(int iter);
    void set_sparse(double sparse);

    void train(cv::Mat const &input);

    void write(std::string const &file) const;
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
        cv::Mat pj_r0_; //same as pj_ expect 0(set to max() of double)
        cv::Mat pj_r1_; //same as pj_ expect 1(set to max() of double)
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

    void generate_activation(layer_struct const &ls,
                             cv::Mat &temp_input);
    void generate_activation_cpu(layer_struct const &ls,
                                 cv::Mat &temp_input);    
    void get_activation(cv::Mat const &input,
                        layer_struct const &es);
    int get_batch_size(int sample_size) const;
    void get_delta_2(cv::Mat const &delta_3,
                     layer_struct const &es);

    void reduce_cost(std::uniform_int_distribution<int> const &uni_int,
                     std::default_random_engine &re, int batch,
                     cv::Mat const &temp_input, layer_struct &ls);    

    void update_weight_and_bias(layer_struct &ls);
    void update_weight_and_bias(cv::Mat const &bias,
                                cv::Mat &weight);

    activation act_;
    cv::Mat activation_;
    int batch_divide_;
    buffer buffer_;
    std::vector<layer_struct> layers_;
    int mat_type_;
    params params_;    
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // AUTOENCODER_H
