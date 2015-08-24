#include "autoencoder.hpp"

#include "../../core/utility.hpp"
#include "../../eigen/cv_to_eigen.hpp"
#include "../../profile/measure.hpp"
#include "propagation.hpp"

#ifdef OCV_HAS_CUDA
#include <opencv2/cudaarithm.hpp>
#endif

#include <Eigen/Dense>

#include <fstream>
#include <random>

namespace ocv{

namespace ml{

namespace{

//opencv do not have an easy way to optimize the codes
//like mat_a = mat_a - constant * mat_b,with eigen
//we can optimize the matrix operation at ease

#define CV2EIGEND(Name, Input) \
    eigen::CV2EigenD Name(reinterpret_cast<double*>(Input.data), \
    Input.rows, \
    Input.cols) \

}

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size) :
    batch_divide_{5},
    mat_type_{CV_64F}
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
 * @brief read the train result from the file(xml)
 * @param file the file save the train result
 */
void autoencoder::read(const std::string &file)
{
    cv::FileStorage in(file, cv::FileStorage::READ);
    int layer_size = 0;
    in["layer_size"]>>layer_size;

    layers_.clear();
    for(int i = 0; i != layer_size; ++i){
        layer_struct ls;
        in["w1_" + std::to_string(i)] >> ls.w1_;
        in["w2_" + std::to_string(i)] >> ls.w2_;
        in["w1_grad_" + std::to_string(i)] >> ls.w1_grad_;
        in["w2_grad_" + std::to_string(i)] >> ls.w2_grad_;
        in["b1_" + std::to_string(i)] >> ls.b1_;
        in["b2_" + std::to_string(i)] >> ls.b2_;
        in["b1_grad_" + std::to_string(i)] >> ls.b1_grad_;
        in["b2_grad_" + std::to_string(i)] >> ls.b2_grad_;
        layers_.emplace_back(ls);
    }
    in["activation"] >> activation_;
}

/**
 * @brief Set the batch divide parameter, this parameter\n
 * will determine the fraction of the samples will be use\n
 * when finding the cost
 * @param fraction train_size = sample_size / fraction,\n
 * the default value is 5
 */
void autoencoder::set_batch_fraction(int fraction)
{
    batch_divide_ = fraction;
}

/**
 * @brief set the beta of autoencoder
 * @param beta the weight of the sparsity penalty term.\n
 * Must be real positive number.The default value is 3
 */
void autoencoder::set_beta(double beta)
{
    params_.beta_ = beta;
}

/**
 * @brief set the hidden layers size
 * @param size size of each hidden layers,must bigger\n
 * than zero.The default value is 0
 */
void autoencoder::
set_hidden_layer_size(cv::AutoBuffer<int> const &size)
{
    params_.hidden_size_ = size;
}

/**
 * @brief set the lambda of the regularization term
 * @param lambda the weight of the regularization term.\n
 * Must be real positive number.The default value is 3e-3
 */
void autoencoder::set_lambda(double lambda)
{
    params_.lambda_ = lambda;
}

/**
 * @brief set the learning rate
 * @param The larger the lrate, the faster we approach the solution,\n
 *  but larger lrate may incurr divergence, must be real\n
 *  positive number.The default value is 2e-2
 */
void autoencoder::set_learning_rate(double lrate)
{
    params_.lrate_ = lrate;
}

/**
 * @brief set maximum iteration time
 * @param iter the maximum iteration time of the algorithm.\n
 * The default value is 80000
 */
void autoencoder::set_max_iter(int iter)
{

    params_.max_iter_ = iter;
}

/**
 * @brief set the sparsity penalty
 * @param sparse Constraint of the hidden neuron, the lower it is,\n
 * the sparser the output of the layer would be.The default\n
 * value is 0.1
 */
void autoencoder::set_sparse(double sparse)
{
    params_.sparse_ = sparse;
}

void autoencoder::generate_activation_impl(layer_struct const &ls,
                                           cv::Mat &temp_input)
{
    activation_.create(ls.w1_.rows, temp_input.cols, mat_type_);
    if(temp_input.cols >= 10000){
#ifdef OCV_HAS_CUDA
        if(cv::cuda::getCudaEnabledDeviceCount() != 0){
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
        }else{
            generate_activation_cpu(ls, temp_input);
        }
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

    CV2EIGEND(eb1, ls.b1_);

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Mapper = Eigen::Map<MatType>;
    Mapper Map(eb1.data(), eb1.size());
    eact.colwise() += Map;
    sigmoid()(activation_);
}

/**
 * @brief train by sparse autoencoder
 * @param input the input image, type must be CV_64F.\n
 * input contains one training example per column
 */
void autoencoder::train(const cv::Mat &input)
{
    CV_Assert(input.type() == CV_64F);

#ifdef OCV_TEST_AUTOENCODER
    test();
#else
    layers_.clear();
    mat_type_ = input.type();
    std::random_device rd;
    std::default_random_engine re(rd());
    int const Batch = get_batch_size(input.cols);
    int const RandomSize = input.cols != Batch ?
                input.cols - Batch - 1 : 0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);
    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        cv::Mat temp_input = i == 0 ? input : activation_;
        layer_struct ls(temp_input.rows, params_.hidden_size_[i],
                        mat_type_);
        reduce_cost(uni_int, re, Batch, temp_input, ls);
        generate_activation(ls, temp_input);
        layers_.push_back(ls);
    }
    act_.clear();
    buffer_.clear();
#endif
}

/**
 * @brief write the training result into the file(xml)
 * @param file the name of the file
 */
void autoencoder::write(const std::string &file) const
{
    cv::FileStorage out(file, cv::FileStorage::WRITE);
    out<<"layer_size"<<(int)layers_.size();
    for(size_t i = 0; i != layers_.size(); ++i){
        auto const &Layer = layers_[i];
        out<<("w1_" + std::to_string(i))<<Layer.w1_;
        out<<("w2_" + std::to_string(i))<<Layer.w2_;
        out<<("w1_grad_" + std::to_string(i))<<Layer.w1_grad_;
        out<<("w2_grad_" + std::to_string(i))<<Layer.w2_grad_;
        out<<("b1_" + std::to_string(i))<<Layer.b1_;
        out<<("b2_" + std::to_string(i))<<Layer.b2_;
        out<<("b1_grad_" + std::to_string(i))<<Layer.b1_grad_;
        out<<("b2_grad_" + std::to_string(i))<<Layer.b2_grad_;
    }

    out<<"activation"<<activation_;
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
    //std::cout<<"square error : "<<SquareError<<"\n";
    // now calculate pj which is the average activation of hidden units
    cv::reduce(act_.hidden_, buffer_.pj_, 1, CV_REDUCE_SUM);
    buffer_.pj_ /= NSamples;

    // the second part is weight decay part
    CV2EIGEND(w1, es.w1_);
    CV2EIGEND(w2, es.w2_);

    double const WeightError =
            ((w1.array() * w1.array()).sum() + (w2.array() * w2.array()).sum()) *
            (params_.lambda_ / 2.0);
    //std::cout<<"weight error : "<<WeightError<<"\n";
    //the third part of overall cost function is the sparsity part
    CV2EIGEND(epj, buffer_.pj_);
    //prevent division by zero
    buffer_.pj_.copyTo(buffer_.pj_r0_);
    buffer_.pj_.copyTo(buffer_.pj_r1_);
    CV2EIGEND(epj_r0, buffer_.pj_r0_);
    CV2EIGEND(epj_r1, buffer_.pj_r1_);
    epj_r0 = (epj.array() != 0.0).
            select(epj, std::numeric_limits<double>::max());
    epj_r1 = (epj.array() != 1.0).
            select(epj, std::numeric_limits<double>::max());//*/

    double const Sparse = params_.sparse_;
    //beta * sum(sparse * log[sparse/pj] +
    //           (1 - sparse) * log[(1-sparse)/(1-pj)])
    double const SparseError =
            ( (Sparse * (Sparse / (epj_r0.array())).log()) +
              (1.0-Sparse)*((1.0-Sparse)/(1.0-epj_r1.array())).log()).sum() *
            params_.beta_;
    //std::cout<<"SparseError error : "<<SparseError<<"\n";
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
    //std::cout<<edelta3<<"\n";
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
    edelta2.noalias() = ew2.transpose() * edelta3;

    buffer_.delta_buffer_.create(buffer_.delta2_.rows, 1, mat_type_);
    CV2EIGEND(epj, buffer_.pj_);
    CV2EIGEND(ebuffer, buffer_.delta_buffer_);

    //Mat temp2 = -sparsityParam / pj + (1 - sparsityParam) / (1 - pj);
    //temp2 *= beta;
    //Mat delta2 = sa.W2.t() * delta3 + repeat(temp2, 1, nsamples);

    CV2EIGEND(epj_r0, buffer_.pj_r0_);
    CV2EIGEND(epj_r1, buffer_.pj_r1_);
    ebuffer = params_.beta_ *
            (-params_.sparse_ / (epj_r0.array()) +
             (1.0 - params_.sparse_) / (1.0 - epj_r1.array()));

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Mapper = Eigen::Map<MatType>;
    Mapper Map(ebuffer.data(), ebuffer.size());
    edelta2.colwise() += Map;

    CV2EIGEND(ehidden, act_.hidden_);
    edelta2 = edelta2.array() * ((1.0 - ehidden.array()) * ehidden.array());//*/
}

void
autoencoder::get_activation(cv::Mat const &input,
                            layer_struct const &es)
{    
    forward_propagation(input, es.w1_, es.b1_, act_.hidden_);
    forward_propagation(act_.hidden_, es.w2_, es.b2_, act_.output_);
}

void autoencoder::generate_activation(const layer_struct &ls,
                                      cv::Mat &temp_input)
{
#ifdef OCV_MEASURE_TIME
    auto const TGen =
            time::measure<>::duration([&]()
    { generate_activation_impl(ls, temp_input); });
    std::cout<<"generate time : "<<TGen.count()<<"\n";
#else
    generate_activation_impl(ls, temp_input);
#endif
}

int autoencoder::get_batch_size(int sample_size) const
{
    if(sample_size > batch_divide_){
        return sample_size / batch_divide_;
    }

    return sample_size;
}

void autoencoder::reduce_cost(const std::uniform_int_distribution<int> &uni_int,
                              std::default_random_engine &re,
                              int batch, const cv::Mat &input,
                              autoencoder::layer_struct &ls)
{
    double last_cost = std::numeric_limits<double>::max();
#ifndef  OCV_MEASURE_TIME
    for(int j = 0; j != params_.max_iter_; ++j){
        auto const ROI = cv::Rect(uni_int(re), 0,
                                  batch, input.rows);
        encoder_cost(input(ROI), ls);
        encoder_gradient(input(ROI), ls);

        if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                ls.cost_ <= 0.0){
            break;
        }

        last_cost = ls.cost_;
        update_weight_and_bias(ls);
    }

#else
    double t_cost = 0;
    double t_gra = 0;
    double t_update = 0;
    int iter_time = 1;
    for(int j = 0; j != params_.max_iter_; ++j){
        auto const ROI = cv::Rect(uni_int(re), 0,
                                  batch, input.rows);
        t_cost +=
                time::measure<>::execution([&]()
        { encoder_cost(input(ROI), ls); });
        t_gra +=
                time::measure<>::execution([&]()
        { encoder_gradient(input(ROI), ls); });
        ++iter_time;

        if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                ls.cost_ <= 0.0){
            break;
        }

        last_cost = ls.cost_;
        t_update +=
                time::measure<>::execution([&]()
        { update_weight_and_bias(ls); });
    }

    std::cout<<"total encoder cost time : "<<t_cost<<"\n";
    std::cout<<"total gradient cost time : "<<t_gra<<"\n";
    std::cout<<"total update time : "<<t_gra<<"\n";
    std::cout<<"average encoder cost time : "<<t_cost / iter_time<<"\n";
    std::cout<<"average gradient cost time : "<<t_gra / iter_time<<"\n";
    std::cout<<"average update time : "<<t_gra / iter_time<<"\n";
#endif
}

void autoencoder::test()
{
#ifdef OCV_TEST_AUTOENCODER
    cv::Mat input;
    cv::FileStorage in("autoencoder_test_data.xml", cv::FileStorage::READ);
    in["train"]>>input;

    layers_.clear();
    mat_type_ = input.type();
    std::random_device rd;
    std::default_random_engine re(rd());

    std::uniform_int_distribution<int>
            uni_int(0, 0);
    cv::Mat hidden_size;
    in["hidden_size"]>>hidden_size;
    params_.hidden_size_.resize(hidden_size.cols);
    for(int i = 0; i != hidden_size.cols; ++i){
        params_.hidden_size_[i] = hidden_size.at<int>(0, i);
    }
    in["max_iter"]>>params_.max_iter_;
    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        cv::Mat temp_input = i == 0 ? input : activation_;
        layer_struct ls;
        auto const Index = std::to_string(i);
        in["w1_" + Index]>>ls.w1_;
        in["w2_" + Index]>>ls.w2_;
        in["b1_" + Index]>>ls.b1_;
        in["b2_" + Index]>>ls.b2_;
        in["w1_grad_" + Index]>>ls.w1_grad_;
        in["w2_grad_" + Index]>>ls.w2_grad_;
        in["b1_grad_" + Index]>>ls.b1_grad_;
        in["b2_grad_" + Index]>>ls.b2_grad_;

        reduce_cost(uni_int, re, input.cols, temp_input, ls);
        generate_activation(ls, temp_input);
        cv::Mat temp_act;
        in["activation_l" + std::to_string(i)] >> temp_act;
        bool all_same = true;
        ocv::for_each_channels<double>(activation_, temp_act,
                                       [&](double lhs, double rhs)
        {
            if(std::abs(lhs - rhs) > 0.02){
                all_same = false;
            }
        });
        if(!all_same){
            break;
        }else{
            std::cout<<"layer "<<i<<" pass\n";
        }
        layers_.push_back(ls);
    }
#endif
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

autoencoder::layer_struct::layer_struct() :
    cost_{0}
{    
    cost_ = 0;
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
    pj_r0_.release();
    pj_r1_.release();
}

void autoencoder::activation::clear()
{
    hidden_.release();
    output_.release();
}

}}

