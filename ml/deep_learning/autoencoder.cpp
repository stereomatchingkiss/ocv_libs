#include "autoencoder.hpp"

#include "core/utility.hpp"
#include "ml/deep_learning/propagation.hpp"

namespace ocv{

namespace ml{

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size)
{
    set_hidden_layer_size(hidden_size);
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

void autoencoder::encoder_cost(const cv::Mat &input,
                               encoder_struct &es)
{
    get_activation(input, es);
    auto const NSamples = input.cols;

    //square error of back propagation(first half)
    cv::subtract(act_.output_, input, sqr_error_);
    cv::pow(sqr_error_, 2.0, sqr_error_);
    sqr_error_ /= 2.0;
    double const SquareError = sum(sqr_error_)[0] / NSamples;

    // now calculate pj which is the average activation of hidden units
    cv::reduce(act_.hidden_, pj_, 1, CV_REDUCE_SUM);
    pj_ /= NSamples;

    // the second part is weight decay part
    cv::pow(es.w1_, 2.0, w1_pow_);
    cv::pow(es.w2_, 2.0, w2_pow_);
    double const WeightError = (cv::sum(w1_pow_)[0] + cv::sum(w2_pow_)[0]) *
            (params_.lambda_ / 2.0);

    //the third part of overall cost function is the sparsity part
    //sparse * log(sparse/pj);
    cv::divide(params_.sparse_, pj_, sparse_error_buffer_);
    cv::log(sparse_error_buffer_, sparse_error_buffer_);
    sparse_error_buffer_ *= params_.sparse_;
    sparse_error_buffer_.copyTo(sparse_error_);

    //(1 - sparse) * log[(1 - sparse)/(1 - pj)]
    cv::subtract(1, pj_, pj_);
    cv::divide(1 - params_.sparse_, pj_, sparse_error_buffer_);
    cv::log(sparse_error_buffer_, sparse_error_buffer_);
    sparse_error_buffer_ *= (1 - params_.sparse_);

    sparse_error_ += sparse_error_buffer_;
    es.cost_ = SquareError + WeightError +
            sum(sparse_error_)[0] * params_.beta_;
}

void
autoencoder::get_activation(cv::Mat const &input,
                            autoencoder::encoder_struct const &es)
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

autoencoder::encoder_struct::
encoder_struct(int input_size, int hidden_size, double cost) :
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

