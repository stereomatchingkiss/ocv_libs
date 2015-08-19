#include "autoencoder.hpp"

#include "core/utility.hpp"

namespace ocv{

namespace ml{

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size)
{
    set_hidden_layer_size(hidden_size);
}

void autoencoder::set_beta(double beta)
{
    params_.beta_ = beta;
}

void autoencoder::
set_hidden_layer_size(cv::AutoBuffer<int> const &size)
{
    params_.hidden_size_ = size;
}

void autoencoder::set_lambda(double lambda)
{
    params_.lambda_ = lambda;
}

void autoencoder::set_learning_rate(double lrate)
{
    params_.lrate_ = lrate;
}

void autoencoder::set_max_iter(int iter)
{
    params_.max_iter_ = iter;
}

void autoencoder::set_sparse(double sparse)
{
    params_.sparse_ = sparse;
}

autoencoder::params::params() :
    beta_(3),
    lambda_(3e-3),
    lrate_(2e-2),
    max_iter_(80000),
    sparse_(0.1)
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

