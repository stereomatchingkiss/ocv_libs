#include "softmax.hpp"

#include "../../ml/utility/gradient_checking.hpp"
#include "../../eigen/eigen.hpp"

#include <iostream>

namespace ocv{

namespace ml{

namespace{

using MatType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Mapper = Eigen::Map<MatType, Eigen::Aligned>;
using MapperConst = Eigen::Map<const MatType, Eigen::Aligned>;

template<typename T>
void print_dimension(T const &input)
{
    std::cout<<input.rows()<<", "<<input.cols()<<"\n\n";
}

}

softmax::softmax()
{

}

const softmax::EigenMat &softmax::get_weight() const
{
    return weight_;
}

void softmax::set_batch_size(int batch_size)
{
    params_.batch_size_ = batch_size;
}

void softmax::set_lambda(double lambda)
{
    params_.lambda_ = lambda;
}

void softmax::set_learning_rate(double lrate)
{
    params_.lrate_ = lrate;
}

void softmax::set_max_iter(size_t max_iter)
{
    params_.max_iter_ = max_iter;
}

void softmax::train(const softmax::EigenMat &train,
                    const std::vector<int> &labels,
                    size_t num_class)
{
    weight_ = EigenMat::Random(num_class, train.rows());
    grad_ = EigenMat::Zero(num_class, train.rows());
    ground_truth_ = EigenMat::Zero(num_class, train.cols());
    for(size_t i = 0; i != ground_truth_.cols(); ++i){
        ground_truth_(labels[i], i) = 1;
    }

#ifdef OCV_TEST_SOFTMAX
    gradient_checking gc;
    auto func = [&](EigenMat &theta)->double
    {
        return compute_cost(train, theta);
    };

    EigenMat const WeightBuffer = weight_;
    EigenMat const Gradient =
            gc.compute_gradient(weight_, func);

    compute_cost(train, WeightBuffer);
    compute_gradient(train);

    std::cout<<std::boolalpha<<"gradient checking pass : "
            <<gc.compare_gradient(grad_, Gradient)<<"\n";//*/
#endif

    for(size_t i = 0; i != params_.max_iter_; ++i){
        auto const Cost = compute_cost(train, weight_);
        if(std::abs(params_.cost_ - Cost) < params_.inaccuracy_){
            break;
        }
        params_.cost_ = Cost;
        compute_gradient(train);
        weight_.array() -= grad_.array() * params_.lrate_;
    }
}

double softmax::compute_cost(EigenMat const &train,
                             EigenMat const &weight)
{
    compute_hypothesis(train, weight);
    double const NSamples = static_cast<double>(train.cols());

    return  -1.0 * (hypothesis_.array().log() *
                    ground_truth_.array()).sum() / NSamples +
            weight.array().pow(2.0).sum() * params_.lambda_ / 2.0;
}

void softmax::compute_hypothesis(EigenMat const &train,
                                 EigenMat const &weight)
{
    hypothesis_.noalias() = weight * train;
    max_exp_power_ = hypothesis_.colwise().maxCoeff();
    for(size_t i = 0; i != hypothesis_.cols(); ++i){
        hypothesis_.col(i).array() -= max_exp_power_(0, i);
    }

    hypothesis_ = hypothesis_.array().exp();
    weight_sum_ = hypothesis_.array().colwise().sum();
    for(size_t i = 0; i != hypothesis_.cols(); ++i){
        hypothesis_.col(i) /= weight_sum_(0, i);
    }
}

void softmax::compute_gradient(const softmax::EigenMat &train)
{
    grad_.noalias() =
            (ground_truth_.array() - hypothesis_.array())
            .matrix() * train.transpose();
    auto const NSamples = static_cast<double>(train.cols());
    grad_.array() /= -NSamples;
}

softmax::params::params() :
    batch_size_{100},
    cost_{std::numeric_limits<double>::max()},
    inaccuracy_{0.002},
    lambda_{0.0},
    lrate_{0.2},
    max_iter_{100}
{

}

}}
