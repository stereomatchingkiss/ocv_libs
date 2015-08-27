#include "softmax.hpp"

#include "../../ml/utility/gradient_checking.hpp"
#include "../../eigen/eigen.hpp"

#include <iostream>
#include <limits>
#include <random>
#include <set>

namespace ocv{

namespace ml{

softmax::softmax()
{

}

/**
 * @brief get the weight of softmax
 * @return weight of softmax
 */
const softmax::EigenMat &softmax::get_weight() const
{
    return weight_;
}

/**
 * @brief Set the batch size of mini-batch
 * @param batch_size batch size of mini-batch,default\n
 * value is 200, if the train data is smaller than\n
 * the batch size, the batch size will be same as the\n
 * batch size
 */
void softmax::set_batch_size(int batch_size)
{
    params_.batch_size_ = batch_size;
}

/**
 * @brief softmax::set_epsillon
 * @param epsillon The desired accuracy or change\n
 *  in parameters at which the iterative algorithm stops.
 */
void softmax::set_epsillon(double epsillon)
{
    params_.epsillon_ = epsillon;
}

/**
 * @brief Setup the lambda
 * @param lambda the lambda value which determine the effect\n
 * of penalizes term
 */
void softmax::set_lambda(double lambda)
{
    params_.lambda_ = lambda;
}

/**
 * @brief Set the learning rate
 * @param lrate The larger the learning rate, the faster\n
 * the convergence speed, but larger value may cause divergence too
 */
void softmax::set_learning_rate(double lrate)
{
    params_.lrate_ = lrate;
}

/**
 * @brief Set max iterateration times
 * @param max_iter max iteration time
 */
void softmax::set_max_iter(int max_iter)
{
    params_.max_iter_ = max_iter;
}

/**
 * @brief read the training result into the data
 * @param file the name of the file
 */
void softmax::read(const std::string &file)
{
    cv::FileStorage in(file, cv::FileStorage::READ);

    in["batch_size"]>>params_.batch_size_;
    in["cost_"]>>params_.cost_;
    in["epsillon_"]>>params_.epsillon_;
    in["lambda_"]>>params_.lambda_;
    in["lrate_"]>>params_.lrate_;
    in["max_iter_"]>>params_.max_iter_;
    cv::Mat weight;
    in["weight"]>>weight;
    eigen::cv2eigen_cpy(weight, weight_);
}

/**
 * @brief Train the input data by softmax algorithm
 * @param train Training data, input contains one\n
 *  training example per column
 * @param labels The label of each training example
 */
void softmax::train(const softmax::EigenMat &train,
                    const std::vector<int> &labels)
{
    auto const UniqueLabels = get_unique_labels(labels);
    auto const NumClass = UniqueLabels.size();
    weight_ = EigenMat::Random(NumClass, train.rows());
    grad_ = EigenMat::Zero(NumClass, train.rows());
    ground_truth_ = EigenMat::Zero(NumClass, train.cols());
    for(size_t i = 0; i != ground_truth_.cols(); ++i){
        auto it = UniqueLabels.find(labels[i]);
        if(it != std::end(UniqueLabels)){
            ground_truth_(it->second, i) = 1;
        }
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

    std::random_device rd;
    std::default_random_engine re(rd());
    auto const TrainCols = static_cast<int>(train.cols());
    int const Batch =
            (get_batch_size(TrainCols));
    int const RandomSize = TrainCols != Batch ?
                TrainCols - Batch - 1 : 0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(size_t i = 0; i != params_.max_iter_; ++i){
        auto const Cols = uni_int(re);
        auto const Cost =
                compute_cost(train.block(0, Cols,
                                         train.rows(),
                                         Batch), weight_);
        if(std::abs(params_.cost_ - Cost) < params_.epsillon_){
            break;
        }
        params_.cost_ = Cost;
        compute_gradient(train.block(0, Cols,
                                     train.rows(),
                                     Batch));
        weight_.array() -= grad_.array() * params_.lrate_;
    }
}

/**
 * @brief write the training result into the file(xml)
 * @param file the name of the file
 */
void softmax::write(const std::string &file) const
{
    cv::FileStorage out(file, cv::FileStorage::WRITE);

    out<<"batch_size"<<params_.batch_size_;
    out<<"cost_"<<params_.cost_;
    out<<"epsillon_"<<params_.epsillon_;
    out<<"lambda_"<<params_.lambda_;
    out<<"lrate_"<<params_.lrate_;
    out<<"max_iter_"<<params_.max_iter_;
    cv::Mat const Weight = eigen::eigen2cv_ref(weight_);
    out<<"weight"<<Weight;
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

int softmax::get_batch_size(int sample_size) const
{
    return std::min(sample_size, params_.batch_size_);
}

std::map<int, int> softmax::
get_unique_labels(const std::vector<int> &labels) const
{
    std::set<int> const UniqueLabels(std::begin(labels),
                                     std::end(labels));
    std::map<int, int> result;
    int i = 0;
    for(auto it = std::begin(UniqueLabels);
        it != std::end(UniqueLabels); ++it){
        if(UniqueLabels.find(*it) ==
                std::end(UniqueLabels)){
            result.emplace(*it, i++);
        }
    }

    return result;
}

softmax::params::params() :
    batch_size_{100},
    cost_{std::numeric_limits<double>::max()},
    epsillon_{0.002},
    lambda_{0.0},
    lrate_{0.2},
    max_iter_{100}
{

}

}}

