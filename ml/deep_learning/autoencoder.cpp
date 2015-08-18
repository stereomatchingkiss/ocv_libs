#include "autoencoder.hpp"

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

}}

