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
    using EigenMat = eigen::MatRowMajor<double>;

    struct layer
    {
        layer();
        layer(int input_size, int hidden_size,
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

    explicit autoencoder(cv::AutoBuffer<int> const &hidden_size);

    autoencoder& operator=(autoencoder const&) = delete;
    autoencoder& operator=(autoencoder &&) = delete;
    autoencoder(autoencoder const&) = delete;
    autoencoder(autoencoder &&) = delete;

    EigenMat const& get_activation() const;
    std::vector<layer> const& get_layer() const;

    void read(std::string const &file);

    void set_batch_size(int size);
    void set_beta(double beta);
    void set_eps(double eps);
    void set_hidden_layer_size(cv::AutoBuffer<int> const &size);
    void set_lambda(double lambda);
    void set_learning_rate(double lrate);
    void set_max_iter(int iter);
    void set_sparse(double sparse);

    void train(EigenMat const &input);

    void write(std::string const &file) const;
private:        
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

    struct cv_layer
    {
        cv_layer();
        cv_layer(int input_size, int hidden_size,
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

    struct criteria
    {
        criteria();

        int batch_size_;
        double beta_;
        double eps_;
        cv::AutoBuffer<int> hidden_size_;
        double lambda_;
        double lrate_; //learning rate
        int max_iter_;
        double sparse_;
    };

    void convert(cv_layer const &input,
                 layer &output) const;
    void convert(layer const &input,
                 cv_layer &output) const;
    void encoder_cost(EigenMat const &input,
                      layer &es);
    void encoder_gradient(EigenMat const &input,
                          layer &es);

    void generate_activation(layer const &ls,
                             EigenMat const &temp_input);
    void generate_activation_cpu(layer const &ls,
                                 EigenMat const &temp_input);
    void generate_activation_impl(layer const &ls,
                                  EigenMat const &temp_input);    
    void get_activation(EigenMat const &input,
                        layer &es);
    int get_batch_size(int sample_size) const;
    void get_delta_2(EigenMat const &delta_3,
                     layer const &es);

    void read_test_data(cv::FileStorage const &in,
                        std::string const &index,
                        cv_layer &out) const;

    void reduce_cost(std::uniform_int_distribution<int> const &uni_int,
                     std::default_random_engine &re,
                     int batch, EigenMat const &input,
                     layer &ls);

    void test();

    void update_weight_and_bias(layer &ls);
    void update_weight_and_bias(EigenMat const &bias,
                                EigenMat &weight);

    activation act_;        
    buffer buffer_;
    criteria params_;
    EigenMat eactivation_;
    std::vector<layer> layers_;        
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // AUTOENCODER_H
