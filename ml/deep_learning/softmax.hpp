#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../ml/utility/gradient_checking.hpp"
#include "../../eigen/eigen.hpp"

#include <opencv2/core.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

/*! \file softmax.hpp
    \brief implement the algorithm--softmax regression based on\n
    the description of UFLDL, these codes are develop based\n
    on the example on the website(http://eric-yuan.me/softmax-regression-cv/#comment-8781).
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

template<typename T = double>
class softmax
{
public:
    static_assert(std::is_floating_point<T>::value,
                  "T should be floating point");

    using EigenMat = eigen::MatRowMajor<T>;

    softmax();

    /**
     * @brief get the weight of softmax
     * @return weight of softmax
     */
    EigenMat const& get_weight() const
    {
        return weight_;
    }

    int predict(Eigen::Ref<const EigenMat> const &input);
    int predict(cv::Mat const &input);

    /**
     * @brief Set the batch size of mini-batch
     * @param batch_size batch size of mini-batch,default\n
     * value is 100, if the train data is smaller than\n
     * the batch size, the batch size will be same as the\n
     * batch size
     */
    void set_batch_size(int batch_size)
    {
        params_.batch_size_ = batch_size;
    }

    /**
     * @brief softmax::set_epsillon
     * @param epsillon The desired accuracy or change\n
     *  in parameters at which the iterative algorithm stops.\n
     *  Default value is 1e-5
     */
    void set_epsillon(double epsillon)
    {
        params_.epsillon_ = epsillon;
    }

    /**
     * @brief Setup the lambda
     * @param lambda the lambda value which determine the effect\n
     * of penalizes term.Default value is 2.0
     */
    void set_lambda(double lambda)
    {
        params_.lambda_ = lambda;
    }

    /**
     * @brief Set the learning rate
     * @param lrate The larger the learning rate, the faster\n
     * the convergence speed, but larger value may cause divergence too.\n
     * Default value is 0.2
     */
    void softmax::set_learning_rate(double lrate)
    {
        params_.lrate_ = lrate;
    }

    /**
     * @brief Set max iterateration times
     * @param max_iter max iteration time, default value is 10000
     */
    void softmax::set_max_iter(int max_iter)
    {
        params_.max_iter_ = max_iter;
    }

    void read(const std::string &file);

    void train(const Eigen::Ref<const EigenMat> &train,
               const std::vector<int> &labels);

    void write(const std::string &file) const;

private:    
    double compute_cost(Eigen::Ref<const EigenMat> const &train,
                        Eigen::Ref<const EigenMat> const &weight,
                        Eigen::Ref<const EigenMat> const &ground_truth);

    void compute_gradient(Eigen::Ref<const EigenMat> const &train,
                          Eigen::Ref<const EigenMat> const &weight,
                          Eigen::Ref<const EigenMat> const &ground_truth);

    void compute_hypothesis(Eigen::Ref<const EigenMat> const &train,
                            Eigen::Ref<const EigenMat> const &weight);

    int get_batch_size(int sample_size) const
    {
        return std::min(sample_size, params_.batch_size_);
    }

    EigenMat get_ground_truth(int NumClass,
                              int samples_size,
                              std::map<int, int> const &unique_labels,
                              std::vector<int> const &labels) const;
    std::map<int, int> softmax::
    get_unique_labels(const std::vector<int> &labels) const;

    void gradient_check(Eigen::Ref<const EigenMat> const &train,
                        Eigen::Ref<const EigenMat> const &ground_truth)
    {
        gradient_checking gc;
        auto func = [&](EigenMat &theta)->double
        {
            return compute_cost(train, theta, ground_truth);
        };

        EigenMat const WeightBuffer = weight_;
        EigenMat const Gradient =
                gc.compute_gradient(weight_, func);

        compute_cost(train, WeightBuffer, ground_truth);
        compute_gradient(train, WeightBuffer, ground_truth);

        std::cout<<std::boolalpha<<"gradient checking pass : "
                <<gc.compare_gradient(grad_, Gradient)<<"\n";//*/
    }


    struct criteria
    {
        criteria();
        int batch_size_;
        double cost_;
        double epsillon_;
        double lambda_;
        double lrate_;
        int max_iter_;
    };

    EigenMat hypothesis_;
    EigenMat grad_;
    EigenMat max_exp_power_;
    criteria params_;
    EigenMat probability_;
    EigenMat weight_;
    EigenMat weight_sum_;
};

template<typename T>
softmax<T>::softmax()
{

}

/**
 *@brief Predicts the response for input sample(one sample)
 *@param input input data for prediction
 *@return Output prediction responses for corresponding sample
 *@pre rows are the features, col is the corresponding sample.\n
 * This function can predict one sample only
 */
template<typename T>
int softmax<T>::predict(Eigen::Ref<const EigenMat> const &input)
{    
    CV_Assert(input.cols() == 1);
    compute_hypothesis(input, weight_);
    probability_ = (hypothesis_ * input.transpose()).
            rowwise().sum();
    EigenMat::Index max_row = 0, max_col = 0;
    probability_.maxCoeff(&max_row, &max_col);

    return max_row;
}

/**
 *@brief Predicts the response for input sample(one sample)
 *@param input input data for prediction
 *@return Output prediction responses for corresponding sample
 *@pre rows are the features, col is the corresponding sample.\n
 * This function can predict one sample only
 */
template<typename T>
int softmax<T>::predict(cv::Mat const &input)
{
    Eigen::Map<EigenMat> const Map(reinterpret_cast<*>(input.data),
                                   input.rows,
                                   input.step / sizeof(T));
    return predict(Map.block(0, 0, input.rows, input.cols));
}

/**
 * @brief read the training result into the data
 * @param file the name of the file
 */
template<typename T>
void softmax<T>::read(const std::string &file)
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
template<typename T>
void softmax<T>::train(const Eigen::Ref<const EigenMat> &train,
                       const std::vector<int> &labels)
{
    auto const UniqueLabels = get_unique_labels(labels);
    auto const NumClass = UniqueLabels.size();
    weight_ = EigenMat::Random(NumClass, train.rows());
    grad_ = EigenMat::Zero(NumClass, train.rows());
    auto const TrainCols = static_cast<int>(train.cols());
    EigenMat const GroundTruth = get_ground_truth(NumClass, TrainCols,
                                                  UniqueLabels,
                                                  labels);
#ifdef OCV_TEST_SOFTMAX
    gradient_check(train, GroundTruth);
#endif

    std::random_device rd;
    std::default_random_engine re(rd());
    int const Batch = (get_batch_size(TrainCols));
    int const RandomSize = TrainCols != Batch ?
                TrainCols - Batch - 1 : 0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(size_t i = 0; i != params_.max_iter_; ++i){
        auto const Cols = uni_int(re);
        auto const &TrainBlock =
                train.block(0, Cols, train.rows(), Batch);
        auto const &GTBlock =
                GroundTruth.block(0, Cols, NumClass, Batch);
        auto const Cost = compute_cost(TrainBlock, weight_, GTBlock);
        if(std::abs(params_.cost_ - Cost) < params_.epsillon_ ||
                Cost < 0){
            break;
        }
        params_.cost_ = Cost;
        compute_gradient(TrainBlock, weight_, GTBlock);
        weight_.array() -= grad_.array() * params_.lrate_;//*/
    }
}

template<typename T>
void softmax<T>::write(const std::string &file) const
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

template<typename T>
double softmax<T>::compute_cost(const Eigen::Ref<const EigenMat> &train,
                                const Eigen::Ref<const EigenMat> &weight,
                                const Eigen::Ref<const EigenMat> &ground_truth)
{    
    compute_hypothesis(train, weight);
    double const NSamples = static_cast<double>(train.cols());
    return  -1.0 * (hypothesis_.array().log() *
                    ground_truth.array()).sum() / NSamples +
            weight.array().pow(2.0).sum() * params_.lambda_ / 2.0;
}

template<typename T>
void softmax<T>::compute_gradient(Eigen::Ref<const EigenMat> const &train,
                                  Eigen::Ref<const EigenMat> const &weight,
                                  Eigen::Ref<const EigenMat> const &ground_truth)
{
    grad_.noalias() =
            (ground_truth.array() - hypothesis_.array())
            .matrix() * train.transpose();
    auto const NSamples = static_cast<double>(train.cols());
    grad_.array() = grad_.array() / -NSamples +
            params_.lambda_ * weight.array();
}

template<typename T>
void softmax<T>::compute_hypothesis(Eigen::Ref<const EigenMat> const &train,
                                    Eigen::Ref<const EigenMat> const &weight)
{    
    hypothesis_.noalias() = weight * train;
    max_exp_power_ = hypothesis_.colwise().maxCoeff();
    for(size_t i = 0; i != hypothesis_.cols(); ++i){
        hypothesis_.col(i).array() -= max_exp_power_(0, i);
    }

    hypothesis_ = hypothesis_.array().exp();
    weight_sum_ = hypothesis_.array().colwise().sum();
    for(size_t i = 0; i != hypothesis_.cols(); ++i){
        if(weight_sum_(0, i) != T(0)){
            hypothesis_.col(i) /= weight_sum_(0, i);
        }        
    }
    hypothesis_ = (hypothesis_.array() != 0 ).
            select(hypothesis_, T(0.1));
}

template<typename T>
typename softmax<T>::EigenMat softmax<T>::
get_ground_truth(int NumClass, int samples_size,
                 std::map<int, int> const &unique_labels,
                 std::vector<int> const &labels) const
{
    EigenMat ground_truth = EigenMat::Zero(NumClass, samples_size);
    for(size_t i = 0; i != ground_truth.cols(); ++i){
        auto it = unique_labels.find(labels[i]);
        if(it != std::end(unique_labels)){
            ground_truth(it->second, i) = 1;
        }
    }

    return ground_truth;
}

template<typename T>
std::map<int, int> softmax<T>::
get_unique_labels(const std::vector<int> &labels) const
{
    std::set<int> const UniqueLabels(std::begin(labels),
                                     std::end(labels));
    std::map<int, int> result;
    int i = 0;
    for(auto it = std::begin(UniqueLabels);
        it != std::end(UniqueLabels); ++it){
        if(result.find(*it) ==
                std::end(result)){
            result.emplace(*it, i++);
        }
    }

    return result;
}

template<typename T>
softmax<T>::criteria::criteria() :
    batch_size_{100},
    cost_{std::numeric_limits<double>::max()},
    epsillon_{1e-5},
    lambda_{2.0},
    lrate_{0.2},
    max_iter_{10000}
{

}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // SOFTMAX_H
