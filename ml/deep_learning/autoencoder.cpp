#include "autoencoder.hpp"

#include "core/utility.hpp"
#include "ml/deep_learning/propagation.hpp"

#include <Eigen/Dense>

#include <random>

namespace ocv{

namespace ml{

namespace{

//opencv do not have an easy way to optimize the codes
//like mat_a = mat_a - constant * mat_b,with eigen
//we can optimize the matrix operation at ease
using namespace Eigen;

template<typename T = double>
using CV2Eigen =
Eigen::Map<Eigen::Matrix<T,
Eigen::Dynamic,
Eigen::Dynamic,Eigen::RowMajor> >;

using CV2EigenD = CV2Eigen<double>;

#define CV2EIGEND(Name, Input) \
    CV2EigenD Name(reinterpret_cast<double*>(Input.data), \
    Input.rows, \
    Input.cols) \

}

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size) :
    mat_type_{CV_32F}
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
    CV_Assert(input.type() == CV_32F || input.type() == CV_64F);

    layers_.clear();
    mat_type_ = input.type();
    double last_cost = std::numeric_limits<double>::max();
    int const MinSize = 1000;
    int const Batch = input.cols >= MinSize ? input.cols / 100 : input.cols;
    int const RandomSize = input.cols >= MinSize ?
                input.cols - input.cols / Batch - 1:
                0;
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        layer_struct ls(input.cols, params_.hidden_size_[i]);
        for(int j = 0; j != params_.max_iter_; ++j){
            auto const ROI = cv::Rect(uni_int(re), 0,
                                      Batch, input.rows);
            encoder_cost(input(ROI), ls);
            encoder_gradient(input(ROI), ls);
            if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                    ls.cost_ <= 0.0){
                break;
            }

            last_cost = ls.cost_;
            update_weight_and_bias(ls.w1_grad_, ls.w1_);
            update_weight_and_bias(ls.w2_grad_, ls.w2_);
            update_weight_and_bias(ls.b1_grad_, ls.b1_);
            update_weight_and_bias(ls.b2_grad_, ls.b2_);
        }
        layers_.push_back(ls);
    }
    act_.clear();
    buffer_.clear();
}

void autoencoder::encoder_cost(const cv::Mat &input,
                               layer_struct &es)
{
    get_activation(input, es);
    auto const NSamples = input.cols;

    //square error of back propagation(first half)
    CV2EIGEND(eout, act_.output_);
    CV2EIGEND(ein, input);
    double const SquareError =
            ((eout.array() - ein.array()).pow(2.0) / 2.0).sum() / NSamples;

    // now calculate pj which is the average activation of hidden units
    cv::reduce(act_.hidden_, buffer_.pj_, 1, CV_REDUCE_SUM);
    buffer_.pj_ /= NSamples;

    // the second part is weight decay part
    CV2EIGEND(w1, es.w1_);
    CV2EIGEND(w2, es.w2_);

    double const WeightError =
            ((w1.array() * w1.array()).sum() + (w2.array() * w2.array()).sum()) *
            (params_.lambda_ / 2.0);

    //the third part of overall cost function is the sparsity part    
    CV2EIGEND(epj, buffer_.pj_);
    double const Sparse = params_.sparse_;

    //beta * sum(sparse * log[sparse/pj] +
    //           (1 - sparse) * log[(1-sparse)/(1-pj)])
    double const SparseError =
            ( (Sparse * (Sparse / epj.array()).log()) +
            (1- Sparse) * ((1 - Sparse) / (1 - epj.array())).log()).sum() *
            params_.beta_;
    es.cost_ = SquareError + WeightError + SparseError;
}

void autoencoder::encoder_gradient(cv::Mat const &input,
                                   layer_struct &es)
{
    auto const NSamples = input.cols;
    buffer_.delta3_.create(input.rows, input.cols, mat_type_);

    CV2EIGEND(edelta3, buffer_.delta3_);
    CV2EIGEND(eact_output, act_.output_);
    CV2EIGEND(einput, input);
    edelta3 = eact_output - einput;
    edelta3 = edelta3.array() *
            ((1.0 - eact_output.array()) * eact_output.array());

    get_delta_2(buffer_.delta3_, es);

    //cv::Mat nablaW1 = delta2 * input.t();
    //cv::Mat nablaW2 = delta3 * act_.hidden_.t();
    //es.w1_grad_ = nablaW1 / NSamples +
    //        params_.lambda_ * es.w1_;

    CV2EIGEND(edelta2, buffer_.delta2_);
    CV2EIGEND(ew1, es.w1_);
    CV2EIGEND(ew2, es.w2_);
    CV2EIGEND(ehidden, act_.hidden_);
    CV2EIGEND(ew1g, es.w1_grad_);
    CV2EIGEND(ew2g, es.w2_grad_);

    ew1g = (edelta2 * einput.transpose()).array() / NSamples +
            params_.lambda_ * ew1.array();
    ew2g = (edelta3 * ehidden.transpose()).array() / NSamples +
            params_.lambda_ * ew2.array();

    cv::reduce(buffer_.delta2_, es.b1_grad_, 1, CV_REDUCE_SUM);
    cv::reduce(buffer_.delta3_, es.b2_grad_, 1, CV_REDUCE_SUM);
    es.b1_grad_ /= NSamples;
    es.b2_grad_ /= NSamples;//*/
}

void autoencoder::get_delta_2(cv::Mat const &delta_3,
                              layer_struct const &es)
{
    //cv::Mat delta2 = es.w2_.t() * delta3 +
    //        cv::repeat(buffer, 1, NSamples);

    buffer_.delta2_.create(es.w2_.cols, delta_3.cols, mat_type_);
    CV2EIGEND(edelta2, buffer_.delta2_);
    CV2EIGEND(edelta3, delta_3);
    CV2EIGEND(ew2, es.w2_);
    edelta2 = ew2.transpose() * edelta3;

    buffer_.delta_buffer_.create(buffer_.delta2_.rows, 1, mat_type_);
    CV2EIGEND(epj, buffer_.pj_);
    CV2EIGEND(ebuffer, buffer_.delta_buffer_);

    ebuffer = params_.beta_ *
            (-params_.sparse_ / epj.array() +
             (1.0 - params_.sparse_) / (1.0 - epj.array()));
    for(int i = 0; i != buffer_.delta2_.cols; ++i){
        buffer_.delta2_.col(i) += buffer_.delta_buffer_;
    }    

    CV2EIGEND(ehidden, act_.hidden_);
    edelta2 = edelta2.array() * ((1.0 - ehidden.array()) * ehidden.array());
}

void
autoencoder::get_activation(cv::Mat const &input,
                            layer_struct const &es)
{
    forward_propagation(input, es.w1_, es.b1_, act_.hidden_);
    forward_propagation(act_.hidden_, es.w2_, es.b2_, act_.output_);
}

void autoencoder::
update_weight_and_bias(const cv::Mat &bias,
                       cv::Mat &weight)
{    
    CV2EIGEND(eweight, weight);
    CV2EIGEND(ebias, bias);

    eweight = eweight.array() -
            params_.lrate_ * ebias.array();
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
    w1_.create(hidden_size, input_size, mat_type_);
    w2_.create(input_size, hidden_size, mat_type_);
    b1_.create(hidden_size, 1, mat_type_);
    b2_.create(input_size, 1, mat_type_);

    generate_random_value<double>(w1_, 0.12);
    generate_random_value<double>(w2_, 0.12);
    generate_random_value<double>(b1_, 0.12);
    generate_random_value<double>(b2_, 0.12);

    w1_grad_ = cv::Mat::zeros(hidden_size, input_size, mat_type_);
    w2_grad_ = cv::Mat::zeros(input_size, hidden_size, mat_type_);
    b1_grad_ = cv::Mat::zeros(hidden_size, 1, mat_type_);
    b2_grad_ = cv::Mat::zeros(input_size, 1, mat_type_);
    cost_ = 0;
}

void autoencoder::buffer::clear()
{
    delta2_.release();
    delta3_.release();
    delta_buffer_.release();
    pj_.release();
}

void autoencoder::activation::clear()
{
    hidden_.release();
    output_.release();
}

}}

