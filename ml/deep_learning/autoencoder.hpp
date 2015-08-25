#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "../../eigen/eigen.hpp"

#include <opencv2/core.hpp>
#include <Eigen/Dense>

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
    using EigenMat = eigen::MatRowMajor<double>;
    //using EigenMat = Eigen::Matrix<double, Eigen::Dynamic,
    //Eigen::Dynamic>;

    struct activation
    {
        void clear();

        EigenMat hidden_;
        EigenMat output_;
    };

    struct buffer
    {
        void clear();

        EigenMat delta2_;
        EigenMat delta3_;
        EigenMat delta_buffer_;
        EigenMat pj_; //the average activation of hidden units
        EigenMat pj_r0_; //same as pj_ expect 0(set to max() of double)
        EigenMat pj_r1_; //same as pj_ expect 1(set to max() of double)
    };

    struct eigen_layer
    {
        eigen_layer();
        eigen_layer(int input_size, int hidden_size,
                    double cost = 0);

        EigenMat w1_;
        EigenMat w2_;
        EigenMat b1_;
        EigenMat b2_;
        EigenMat w1_grad_;
        EigenMat w2_grad_;
        EigenMat b1_grad_;
        EigenMat b2_grad_;
        double cost_;
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

    void convert(layer_struct const &input,
                 eigen_layer &output) const;
    void convert(eigen_layer const &input,
                 layer_struct &output) const;
    void encoder_cost(EigenMat const &input,
                      eigen_layer &es);
    void encoder_gradient(EigenMat const &input,
                          eigen_layer &es);

    void get_activation(EigenMat const &input,
                        eigen_layer &es);
    void generate_activation(eigen_layer const &ls,
                             EigenMat &temp_input);
    void generate_activation_cpu(eigen_layer const &ls,
                                 EigenMat &temp_input);
    void generate_activation_impl(eigen_layer const &ls,
                                  EigenMat &temp_input);
    int get_batch_size(int sample_size) const;
    void get_delta_2(EigenMat const &delta_3,
                     eigen_layer const &es);

    void read_test_data(cv::FileStorage const &in,
                        std::string const &index,
                        layer_struct &out) const;

    void reduce_cost(std::uniform_int_distribution<int> const &uni_int,
                     std::default_random_engine &re,
                     int batch, EigenMat const &input,
                     eigen_layer &ls);

    void test();

    void update_weight_and_bias(eigen_layer &ls);
    void update_weight_and_bias(EigenMat const &bias,
                                EigenMat &weight);

    activation act_;
    cv::Mat activation_;
    int batch_divide_;
    buffer buffer_;
    EigenMat eactivation_;
    std::vector<layer_struct> layers_;
    int mat_type_;
    params params_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // AUTOENCODER_H
