#include "autoencoder.hpp"

#include "core/utility.hpp"
#include "ml/deep_learning/propagation.hpp"

#include <random>

namespace ocv{

namespace ml{

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size)
{
    set_hidden_layer_size(hidden_size);
}

const std::vector<autoencoder::layer_struct> &
autoencoder::get_layer_struct() const
{
    return layers_;
}

/**
 * @brief set the beta of autoencoder
 * @param beta the weight of the sparsity penalty term.\n
 * Must be real positive number
 */
void autoencoder::set_beta(double beta)
{
    params_.beta_ = beta;
}

/**
 * @brief set the hidden layers size
 * @param size size of each hidden layers,must bigger\n
 * than zero
 */
void autoencoder::
set_hidden_layer_size(cv::AutoBuffer<int> const &size)
{
    params_.hidden_size_ = size;
}

/**
 * @brief set the lambda of the regularization term
 * @param lambda the weight of the regularization term.\n
 * Must be real positive number
 */
void autoencoder::set_lambda(double lambda)
{
    params_.lambda_ = lambda;
}

/**
 * @brief set the learning rate
 * @param The larger the lrate, the faster we approach the solution,\n
 *  but larger lrate may incurr divergence, must be real\n
 *  positive number
 */
void autoencoder::set_learning_rate(double lrate)
{
    params_.lrate_ = lrate;
}

/**
 * @brief set maximum iteration time
 * @param iter the maximum iteration time of the algorithm
 */
void autoencoder::set_max_iter(int iter)
{
    params_.max_iter_ = iter;
}

/**
 * @brief set the sparsity penalty
 * @param sparse Constraint of the hidden neuron, the lower it is,\n
 * the sparser the output of the layer would be
 */
void autoencoder::set_sparse(double sparse)
{
    params_.sparse_ = sparse;
}

/**
 * @brief autoencoder::train
 * @param input
 */
void autoencoder::train(const cv::Mat &input)
{
    double last_cost = std::numeric_limits<double>::max();
    std::random_device rd;
    std::default_random_engine re(rd());
    int const MinSize = 1000;
    int const Batch = input.cols >= MinSize ? input.cols / 100 : input.cols;
    int const RandomSize = input.cols >= MinSize ?
                input.cols - input.cols / Batch - 1:
                0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(int i = 0; i < params_.hidden_size_.size(); ++i)
    {
        layer_struct ls(input.cols, params_.hidden_size_[i]);
        for(int j = 0; j != params_.max_iter_; ++j)
        {
            auto const ROI = cv::Rect(uni_int(re), 0,
                                      Batch, input.rows);
            encoder_cost(input(ROI), ls);
            encoder_gradient(input(ROI), ls);
            if(std::abs(last_cost - ls.cost_) < params_.eps_){
                break;
            }
        }
        layers_.push_back(ls);
    }
}

void autoencoder::encoder_cost(const cv::Mat &input,
                               layer_struct &es)
{
    get_activation(input, es);
    auto const NSamples = input.cols;

    //square error of back propagation(first half)
    //sqr_error = sum(pow((output - input), 2) / 2.0） / NSamples
    cv::subtract(act_.output_, input, buffer_.sqr_error_);
    cv::pow(buffer_.sqr_error_, 2.0, buffer_.sqr_error_);
    buffer_.sqr_error_ /= 2.0;
    double const SquareError = sum(buffer_.sqr_error_)[0] / NSamples;

    // now calculate pj which is the average activation of hidden units
    cv::reduce(act_.hidden_, buffer_.pj_, 1, CV_REDUCE_SUM);
    buffer_.pj_ /= NSamples;

    // the second part is weight decay part
    //cv::pow(es.w1_, 2.0, w1_pow_);
    //cv::pow(es.w2_, 2.0, w2_pow_);
    cv::multiply(es.w1_, es.w1_, buffer_.w1_pow_);
    cv::multiply(es.w2_, es.w2_, buffer_.w2_pow_);
    double const WeightError =
            (cv::sum(buffer_.w1_pow_)[0] + cv::sum(buffer_.w2_pow_)[0]) *
            (params_.lambda_ / 2.0);

    //the third part of overall cost function is the sparsity part
    //sparse * log(sparse/pj);
    cv::divide(params_.sparse_, buffer_.pj_,
               buffer_.sparse_error_buffer_);
    cv::log(buffer_.sparse_error_buffer_,
            buffer_.sparse_error_buffer_);
    buffer_.sparse_error_buffer_ *= params_.sparse_;
    buffer_.sparse_error_buffer_.copyTo(buffer_.sparse_error_);

    //(1 - sparse) * log[(1 - sparse)/(1 - pj)]
    cv::divide(1 - params_.sparse_, 1 - buffer_.pj_,
               buffer_.sparse_error_buffer_);
    cv::log(buffer_.sparse_error_buffer_, buffer_.sparse_error_buffer_);
    buffer_.sparse_error_buffer_ *= (1 - params_.sparse_);

    //sparse * log(sparse/pj) + (1 - sparse) * log[(1 - sparse)/(1 - pj)]
    buffer_.sparse_error_ += buffer_.sparse_error_buffer_;
    es.cost_ = SquareError + WeightError +
            sum(buffer_.sparse_error_)[0] * params_.beta_;
}

void autoencoder::encoder_gradient(cv::Mat const &input,
                                   layer_struct &es)
{
    auto const NSamples = input.cols;
    cv::Mat delta3 = act_.output_ - input;
    cv::multiply(delta3, dsigmoid_func(act_.output_), delta3);

    cv::Mat temp2 = -params_.sparse_ / buffer_.pj_ +
            (1.0 - params_.sparse_) / (1.0 - buffer_.pj_);
    temp2 *= params_.beta_;
    //cv::Mat delta2 = es.w2_.t() * delta3 +
    //        cv::repeat(temp2, 1, NSamples);
    cv::Mat delta2 = es.w2_.t() * delta3;
    for(int i = 0; i != delta2.cols; ++i){
        delta2.col(i) += temp2;
    }
    cv::multiply(delta2, dsigmoid_func(act_.hidden_), delta2);

    cv::Mat nablaW1 = delta2 * input.t();
    cv::Mat nablaW2 = delta3 * act_.hidden_.t();
    cv::Mat nablab1 = delta2;
    cv::Mat nablab2 = delta3;
    es.w1_grad_ = nablaW1 / NSamples +
            params_.lambda_ * es.w1_;
    es.w2_grad_ = nablaW2 / NSamples +
            params_.lambda_ * es.w2_;
    cv::reduce(nablab1, es.b1_grad_, 1, CV_REDUCE_SUM);
    cv::reduce(nablab2, es.b2_grad_, 1, CV_REDUCE_SUM);
    es.b1_grad_ /= NSamples;
    es.b2_grad_ /= NSamples;//*/
}

void
autoencoder::get_activation(cv::Mat const &input,
                            layer_struct const &es)
{
    forward_propagation(input, es.w1_, es.b1_, act_.hidden_);
    forward_propagation(act_.hidden_, es.w2_, es.b2_, act_.output_);
}

autoencoder::params::params() :
    beta_{3},
    eps_{5e-5},
    lambda_{3e-3},
    lrate_{2e-2},
    max_iter_{80000},
    sparse_{0.1}
{

}

autoencoder::layer_struct::
layer_struct(int input_size, int hidden_size, double cost) :
    cost_(cost)
{
    w1_.create(hidden_size, input_size, CV_64F);
    w2_.create(input_size, hidden_size, CV_64F);
    b1_.create(hidden_size, 1, CV_64F);
    b2_.create(input_size, 1, CV_64F);

    generate_random_value<double>(w1_, 0.12);
    generate_random_value<double>(w2_, 0.12);
    generate_random_value<double>(b1_, 0.12);
    generate_random_value<double>(b2_, 0.12);

    w1_grad_ = cv::Mat::zeros(hidden_size, input_size, CV_64F);
    w2_grad_ = cv::Mat::zeros(input_size, hidden_size, CV_64F);
    b1_grad_ = cv::Mat::zeros(hidden_size, 1, CV_64F);
    b2_grad_ = cv::Mat::zeros(input_size, 1, CV_64F);
    cost_ = 0;
}

}}

