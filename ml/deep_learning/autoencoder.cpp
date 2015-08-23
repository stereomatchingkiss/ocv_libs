#include "autoencoder.hpp"

#include "../../core/utility.hpp"
#include "../../profile/measure.hpp"
#include "propagation.hpp"

#include <opencv2/cudaarithm.hpp>

#include <Eigen/Dense>

#include <fstream>
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
    mat_type_{CV_32F},
    zero_firewall_{0.02}
{
    set_hidden_layer_size(hidden_size);
}

const cv::Mat &autoencoder::get_activation() const
{
    return activation_;
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

void autoencoder::generate_activation(layer_struct const &ls,
                                      cv::Mat &temp_input)
{
    activation_.create(ls.w1_.rows, temp_input.cols, mat_type_);
    if(temp_input.cols >= 10000 &&
            cv::cuda::getCudaEnabledDeviceCount() != 0){
#ifdef OCV_HAS_CUDA
        cv::cuda::GpuMat cu_w1(ls.w1_);
        cv::cuda::GpuMat cu_in(temp_input);
        cv::cuda::GpuMat cu_act;
        cv::cuda::gemm(cu_w1, cu_in, 1.0, cv::cuda::GpuMat(),
                       0.0, cu_act);
        cu_act.download(activation_);
        for(int i = 0; i != activation_.cols; ++i){
            activation_.col(i) += ls.b1_;
        }
        cu_act.upload(activation_);
        cu_act.convertTo(cu_act, CV_64F, -1.0);
        cv::cuda::exp(cu_act, cu_act);
        cv::cuda::add(cu_act, 1.0, cu_act);
        cv::cuda::divide(1.0, cu_act, cu_act);
        cu_act.download(activation_);
#else
        generate_activation_cpu(ls, temp_input);
#endif
    }else{
        generate_activation_cpu(ls, temp_input);
    }
}

void autoencoder::
generate_activation_cpu(layer_struct const &ls,
                        cv::Mat &temp_input)
{
    CV2EIGEND(eact, activation_);
    CV2EIGEND(ew1, ls.w1_);
    CV2EIGEND(etemp_input, temp_input);
    eact.noalias() = ew1 * etemp_input;
    for(int i = 0; i != activation_.cols; ++i){
        activation_.col(i) += ls.b1_;
    }
    eact = 1.0 / (1.0 + (-1.0 * eact.array()).exp());//*/
}

void autoencoder::train(const cv::Mat &input)
{
    CV_Assert(input.type() == CV_64F);

    layers_.clear();
    mat_type_ = input.type();
    double last_cost = std::numeric_limits<double>::max();
    int const MinSize = 1000;
    std::random_device rd;
    std::default_random_engine re(rd());
    int const Batch = input.cols >= MinSize ?
                input.cols / 100 : input.cols;
    int const RandomSize = input.cols >= MinSize ?
                input.cols - Batch - 1:
                0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        cv::Mat temp_input = i == 0 ? input : activation_;
        layer_struct ls(temp_input.rows, params_.hidden_size_[i],
                        mat_type_);
        for(int j = 0; j != params_.max_iter_; ++j){
            auto const ROI = cv::Rect(uni_int(re), 0,
                                      Batch, temp_input.rows);
            //auto tcost =
            //        time::measure<>::duration([&](){ encoder_cost(temp_input(ROI), ls); });
            //auto tgra =
            //        time::measure<>::duration([&](){ encoder_gradient(temp_input(ROI), ls); });
            //std::cout<<"encoder cost time : "<<tcost.count()<<"\n";
            //std::cout<<"gradient cost time : "<<tgra.count()<<"\n";
            encoder_cost(temp_input(ROI), ls);
            encoder_gradient(temp_input(ROI), ls);
            if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                    ls.cost_ <= 0.0){
                break;
            }

            last_cost = ls.cost_;
            update_weight_and_bias(ls);
            //auto tupdate =
            //        time::measure<>::duration([&](){ update_weight_and_bias(ls); });
            //std::cout<<"update time : "<<tupdate.count()<<"\n";
        }
        generate_activation(ls, temp_input);
        //auto tgen =
        //        time::measure<>::duration([&]()
        //{ generate_activation(ls, temp_input); });
        //std::cout<<"generate time : "<<tgen.count()<<"\n";
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
            ( (Sparse * (Sparse / epj.array() + zero_firewall_).log()) +
              (1- Sparse) * ((1 - Sparse) / (1 - epj.array() + zero_firewall_)).log()).sum() *
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
    edelta3.noalias() = eact_output - einput;
    edelta3 = edelta3.array() *
            ((1.0 - eact_output.array()) * eact_output.array());

    get_delta_2(buffer_.delta3_, es);

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
    es.b2_grad_ /= NSamples;
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
    edelta2.noalias() = ew2.transpose() * edelta3;

    buffer_.delta_buffer_.create(buffer_.delta2_.rows, 1, mat_type_);
    CV2EIGEND(epj, buffer_.pj_);
    CV2EIGEND(ebuffer, buffer_.delta_buffer_);

    //Mat temp2 = -sparsityParam / pj + (1 - sparsityParam) / (1 - pj);
    //temp2 *= beta;
    //Mat delta2 = sa.W2.t() * delta3 + repeat(temp2, 1, nsamples);

    ebuffer = params_.beta_ *
            (-params_.sparse_ / (epj.array() + zero_firewall_) +
             (1.0 - params_.sparse_) / (1.0 - epj.array() + zero_firewall_));
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

void autoencoder::update_weight_and_bias(layer_struct &ls)
{
    update_weight_and_bias(ls.w1_grad_, ls.w1_);
    update_weight_and_bias(ls.w2_grad_, ls.w2_);
    update_weight_and_bias(ls.b1_grad_, ls.b1_);
    update_weight_and_bias(ls.b2_grad_, ls.b2_);
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
layer_struct(int input_size, int hidden_size,
             int mat_type, double cost) :
    cost_(cost)
{
    w1_.create(hidden_size, input_size, mat_type);
    w2_.create(input_size, hidden_size, mat_type);
    b1_.create(hidden_size, 1, mat_type);
    b2_.create(input_size, 1, mat_type);

    generate_random_value<double>(w1_, 0.12);
    generate_random_value<double>(w2_, 0.12);
    generate_random_value<double>(b1_, 0.12);
    generate_random_value<double>(b2_, 0.12);

    w1_grad_ = cv::Mat::zeros(hidden_size, input_size, mat_type);
    w2_grad_ = cv::Mat::zeros(input_size, hidden_size, mat_type);
    b1_grad_ = cv::Mat::zeros(hidden_size, 1, mat_type);
    b2_grad_ = cv::Mat::zeros(input_size, 1, mat_type);
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

