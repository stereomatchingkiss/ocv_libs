#include "autoencoder.hpp"

#include "../../core/utility.hpp"
#include "../../eigen/eigen.hpp"
#include "../../profile/measure.hpp"
#include "propagation.hpp"

#include <opencv2/core/eigen.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <random>

namespace ocv{

namespace ml{

namespace{

using MatType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Mapper = Eigen::Map<MatType, Eigen::Aligned>;
using MapperConst = Eigen::Map<const MatType, Eigen::Aligned>;

template <typename T>
void unused(T &&)
{ };

}

autoencoder::autoencoder(cv::AutoBuffer<int> const &hidden_size)
{    
    set_hidden_layer_size(hidden_size);
}

autoencoder::EigenMat const &autoencoder::
get_activation() const
{
    return eactivation_;
}

const std::vector<autoencoder::layer> &
autoencoder::get_layer() const
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
    in["batch_size"]>>params_.batch_size_;
    in["beta_"]>>params_.beta_;
    in["eps_"]>>params_.eps_;
    in["lambda_"]>>params_.lambda_;
    in["lrate_"]>>params_.lrate_;
    in["max_iter_"]>>params_.max_iter_;
    in["sparse_"]>>params_.sparse_;
    params_.hidden_size_.resize(layer_size);
    layers_.clear();
    for(int i = 0; i != layer_size; ++i){
        cv_layer ls;
        auto const Index = std::to_string(i);
        in["w1_" + Index] >> ls.w1_;
        in["w2_" + Index] >> ls.w2_;
        in["w1_grad_" + Index] >> ls.w1_grad_;
        in["w2_grad_" + Index] >> ls.w2_grad_;
        in["b1_" + Index] >> ls.b1_;
        in["b2_" + Index] >> ls.b2_;
        in["b1_grad_" + Index] >> ls.b1_grad_;
        in["b2_grad_" + Index] >> ls.b2_grad_;
        in["hidden_size_" + Index]>>
             params_.hidden_size_[i];
        layer el;
        convert(ls, el);
        layers_.emplace_back(el);
    }
    cv::Mat activation;
    in["activation"] >> activation;
    eigen::cv2eigen_cpy(activation, eactivation_);
}

/**
 * @brief Set the batch divide parameter, this parameter\n
 * will determine the fraction of the samples will be use\n
 * when finding the cost
 * @param size set up the mini-batch size every time the\n
 * iteration will use
 */
void autoencoder::set_batch_size(int size)
{
    params_.batch_size_ = size;
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

void autoencoder::generate_activation_impl(layer const &ls,
                                           EigenMat const &temp_input)
{    
    generate_activation_cpu(ls, temp_input);
}

void autoencoder::
generate_activation_cpu(layer const &ls,
                        EigenMat const &temp_input)
{        
    if(temp_input.data() != eactivation_.data()){
        eactivation_.noalias() = ls.w1_ * temp_input;
    }else{
        eactivation_ = ls.w1_ * temp_input;
    }

    MapperConst Map(ls.b1_.data(), ls.b1_.size());
    eactivation_.colwise() += Map;
    sigmoid()(eactivation_);
}

/**
 * @brief train by sparse autoencoder
 * @param input the input image, type must be double.\n
 * input contains one training example per column
 */

/*! \brief example.
 *\code
 * ocv::eigen::EigenMat buffer(16*16, 10000);
 * cv::Mat train = ocv::eigen::eigen2cv_ref(buffer);
 * //read_mnist will read the data of mnist into cv::Mat
 * read_mnist(train, "train_0", 10000);
 * train /= 255.0;
 *
 * cv::AutoBuffer<int> hidden_size(2);
 * hidden_size[0] = buffer.cols() * 3 / 4;
 * hidden_size[1] = hidden_size[0];
 * ocv::ml::autoencoder encoder(hidden_size);
 * encoder.set_batch_fraction(20);
 * encoder.train(train);
 * encoder.write("train.xml");
 *\endcode
*/
void autoencoder::train(const EigenMat &input)
{    
#ifdef OCV_TEST_AUTOENCODER
    unused(input);
    test();
#else    
    layers_.clear();
    std::random_device rd;
    std::default_random_engine re(rd());
    int const Batch = get_batch_size(input.cols());
    int const RandomSize = input.cols() != Batch ?
                input.cols() - Batch - 1 : 0;
    std::uniform_int_distribution<int>
            uni_int(0, RandomSize);

    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        EigenMat const &temp_input = i == 0 ? input
                                            : eactivation_;
        layer es(temp_input.rows(), params_.hidden_size_[i]);
        reduce_cost(uni_int, re, Batch, temp_input, es);
        generate_activation(es, temp_input);
        layers_.push_back(es);
    }
    act_.clear();
    buffer_.clear();//*/
#endif
}

/**
 * @brief write the training result into the file(xml)
 * @param file the name of the file
 */
void autoencoder::write(const std::string &file) const
{
    cv::FileStorage out(file, cv::FileStorage::WRITE);
    out<<"layer_size"<<static_cast<int>(layers_.size());
    out<<"batch_size"<<params_.batch_size_;
    out<<"beta_"<<params_.beta_;
    out<<"eps_"<<params_.eps_;
    out<<"lambda_"<<params_.lambda_;
    out<<"lrate_"<<params_.lrate_;
    out<<"max_iter_"<<params_.max_iter_;
    out<<"sparse_"<<params_.sparse_;
    for(size_t i = 0; i != layers_.size(); ++i){
        cv_layer ls;
        convert(layers_[i], ls);
        auto const Index = std::to_string(i);
        out<<("w1_" + Index)<<ls.w1_;
        out<<("w2_" + Index)<<ls.w2_;
        out<<("w1_grad_" + Index)<<ls.w1_grad_;
        out<<("w2_grad_" + Index)<<ls.w2_grad_;
        out<<("b1_" + Index)<<ls.b1_;
        out<<("b2_" + Index)<<ls.b2_;
        out<<("b1_grad_" + Index)<<ls.b1_grad_;
        out<<("b2_grad_" + Index)<<ls.b2_grad_;
        out<<("hidden_size_" + Index)<<
             params_.hidden_size_[i];
    }
    cv::Mat const Activation = eigen::eigen2cv_ref(eactivation_);
    out<<"activation"<<Activation;
}

void autoencoder::convert(cv_layer const &input,
                          layer &output) const
{        
    eigen::cv2eigen_cpy(input.b1_, output.b1_);
    eigen::cv2eigen_cpy(input.b2_, output.b2_);
    eigen::cv2eigen_cpy(input.w1_, output.w1_);
    eigen::cv2eigen_cpy(input.w2_, output.w2_);
    eigen::cv2eigen_cpy(input.b1_grad_, output.b1_grad_);
    eigen::cv2eigen_cpy(input.b2_grad_, output.b2_grad_);
    eigen::cv2eigen_cpy(input.w1_grad_, output.w1_grad_);
    eigen::cv2eigen_cpy(input.w2_grad_, output.w2_grad_);
}

void autoencoder::convert(const layer &input,
                          cv_layer &output) const
{
    eigen::eigen2cv_cpy(input.b1_, output.b1_);
    eigen::eigen2cv_cpy(input.b2_, output.b2_);
    eigen::eigen2cv_cpy(input.w1_, output.w1_);
    eigen::eigen2cv_cpy(input.w2_, output.w2_);
    eigen::eigen2cv_cpy(input.b1_grad_, output.b1_grad_);
    eigen::eigen2cv_cpy(input.b2_grad_, output.b2_grad_);
    eigen::eigen2cv_cpy(input.w1_grad_, output.w1_grad_);
    eigen::eigen2cv_cpy(input.w2_grad_, output.w2_grad_);
}

void autoencoder::encoder_cost(EigenMat const &input,
                               layer &es)
{    
    get_activation(input, es);
    //std::cout<<"get activation\n";
    auto const NSamples = input.cols();
    //square error of back propagation(first half)
    double const SquareError =
            ((act_.output_.array() - input.array()).pow(2.0)
             / 2.0).sum() / NSamples;
    //std::cout<<"square error : "<<SquareError<<"\n";
    // now calculate pj which is the average activation of hidden units
    buffer_.pj_ = act_.hidden_.rowwise().sum() / NSamples;

    // the second part is weight decay part
    double const WeightError =
            ((es.w1_.array() * es.w2_.array()).sum() +
             (es.w2_.array() * es.w2_.array()).sum()) *
            (params_.lambda_ / 2.0);
    //std::cout<<"weight error : "<<WeightError<<"\n";
    //prevent division by zero
    buffer_.pj_r0_ = (buffer_.pj_.array() != 0.0).
            select(buffer_.pj_, std::numeric_limits<double>::max());
    buffer_.pj_r1_ = (buffer_.pj_.array() != 1.0).
            select(buffer_.pj_, std::numeric_limits<double>::max());//*/

    //the third part of overall cost function is the sparsity part
    double const Sparse = params_.sparse_;
    //beta * sum(sparse * log[sparse/pj] +
    //           (1 - sparse) * log[(1-sparse)/(1-pj)])
    double const SparseError =
            ( (Sparse * (Sparse / (buffer_.pj_r0_.array())).log()) +
              (1.0-Sparse)*((1.0-Sparse)/(1.0-buffer_.pj_r1_.array())).log()).sum() *
            params_.beta_;
    //std::cout<<"SparseError error : "<<SparseError<<"\n";
    es.cost_ = SquareError + WeightError + SparseError;//*/
}

void autoencoder::encoder_gradient(EigenMat const &input,
                                   layer &es)
{            
    get_delta_2(buffer_.delta3_, es);

    buffer_.delta3_ = act_.output_ - input;
    buffer_.delta3_ = buffer_.delta3_.array() *
            ((1.0 - act_.output_.array()) * act_.output_.array());

    auto const NSamples = input.cols();
    es.w1_grad_.noalias() = buffer_.delta2_*input.transpose();
    es.w1_grad_ = (es.w1_grad_.array()/NSamples) +
            params_.lambda_ * es.w1_.array();
    es.w2_grad_.noalias() = buffer_.delta3_*act_.hidden_.transpose();
    es.w2_grad_ = (es.w2_grad_.array()/NSamples) +
            params_.lambda_ * es.w2_.array();

    es.b1_grad_ = buffer_.delta2_.rowwise().sum() / NSamples;
    es.b2_grad_ = buffer_.delta3_.rowwise().sum() / NSamples;
}

void autoencoder::get_delta_2(EigenMat const &delta_3,
                              layer const &es)
{
    //cv::Mat delta2 = es.w2_.t() * delta3 +
    //        cv::repeat(buffer, 1, NSamples);

    buffer_.delta2_.noalias() = es.w2_.transpose() * delta_3;
    buffer_.delta_buffer_ =
            params_.beta_ *
            (-params_.sparse_/(buffer_.pj_r0_.array()) +
             (1.0-params_.sparse_)/(1.0-buffer_.pj_r1_.array()));

    Mapper Map(buffer_.delta_buffer_.data(),
               buffer_.delta_buffer_.size());
    buffer_.delta2_.colwise() += Map;
    buffer_.delta2_ =
            buffer_.delta2_.array() *
            ((1.0 - act_.hidden_.array()) * act_.hidden_.array());
}

void autoencoder::read_test_data(cv::FileStorage const &in,
                                 std::string const &index,
                                 cv_layer &out) const
{
#ifdef OCV_TEST_AUTOENCODER
    in["w1_" + index]>>out.w1_;
    in["w2_" + index]>>out.w2_;
    in["b1_" + index]>>out.b1_;
    in["b2_" + index]>>out.b2_;
    in["w1_grad_" + index]>>out.w1_grad_;
    in["w2_grad_" + index]>>out.w2_grad_;
    in["b1_grad_" + index]>>out.b1_grad_;
    in["b2_grad_" + index]>>out.b2_grad_;
#else
    unused(in);
    unused(index);
    unused(out);
#endif
}

void
autoencoder::get_activation(EigenMat const &input,
                            layer &es)
{        
    forward_propagation(input, es.w1_, es.b1_, act_.hidden_);
    forward_propagation(act_.hidden_, es.w2_, es.b2_, act_.output_);
}

void autoencoder::generate_activation(layer const &ls,
                                      EigenMat const &temp_input)
{
#ifdef OCV_MEASURE_TIME
    auto const TGen =
            time::measure<>::duration([&]()
    { generate_activation_impl(ls, temp_input); });
    std::cout<<"time of generate last layer activation : "<<TGen.count()<<"\n\n";
#else
    generate_activation_impl(ls, temp_input);
#endif
}

int autoencoder::get_batch_size(int batch_size) const
{    
    return std::min(params_.batch_size_, batch_size);
}

void autoencoder::reduce_cost(std::uniform_int_distribution<int> const &uni_int,
                              std::default_random_engine &re,
                              int batch, EigenMat const &input,
                              layer &ls)
{
    double last_cost = std::numeric_limits<double>::max();
    auto const LRate = params_.lrate_;
#ifndef  OCV_MEASURE_TIME
    for(int j = 0; j != params_.max_iter_; ++j){
        int const X = uni_int(re);
        encoder_cost(input.block(0, X,
                                 input.rows(), batch), ls);
        if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                ls.cost_ <= 0.0){
            break;
        }

        encoder_gradient(input.block(0, X,
                                     input.rows(), batch), ls);
        if(ls.cost_ > last_cost){
            params_.lrate_ /= 2;
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
        int const X = uni_int(re);
        t_cost +=
                time::measure<>::execution([&]()
        { encoder_cost(input.block(0, X,
                                   input.rows(), batch), ls); });
        t_gra +=
                time::measure<>::execution([&]()
        { encoder_gradient(input.block(0, X,
                                       input.rows(), batch), ls); });
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
    std::cout<<"total time of update weight and bias : "<<t_update<<"\n";
    std::cout<<"average encoder cost time : "<<t_cost / iter_time<<"\n";
    std::cout<<"average gradient cost time : "<<t_gra / iter_time<<"\n";
    std::cout<<"average update time : "<<t_gra / iter_time<<"\n";
    std::cout<<"average time of update weight and bias : "<<t_update / iter_time<<"\n";
#endif
    params_.lrate_ = LRate;
}

void autoencoder::test()
{
#ifdef OCV_TEST_AUTOENCODER    
    cv::Mat input;
    cv::FileStorage in("autoencoder_test_data.xml", cv::FileStorage::READ);
    in["train"]>>input;

    EigenMat ein;
    eigen::cv2eigen_cpy(input, ein);

    layers_.clear();
    std::random_device rd;
    std::default_random_engine re(rd());

    std::uniform_int_distribution<int>
            uni_int(0, 0);
    cv::Mat hidden_size;
    in["hidden_size"]>>hidden_size;
    //std::cout<<"hidden size : "<<hidden_size<<"\n";
    params_.hidden_size_.resize(hidden_size.cols);
    for(int i = 0; i != hidden_size.cols; ++i){
        params_.hidden_size_[i] = hidden_size.at<int>(0, i);
    }
    in["max_iter"]>>params_.max_iter_;
    //std::cout<<"max iter : "<<params_.max_iter_<<"\n";
    for(int i = 0; i < params_.hidden_size_.size(); ++i){
        EigenMat &temp_input = i == 0 ? ein : eactivation_;
        cv_layer ls;
        read_test_data(in, std::to_string(i),
                       ls);
        layer es;
        convert(ls, es);

        reduce_cost(uni_int, re, temp_input.cols(),
                    temp_input, es);
        generate_activation(es, temp_input);
        cv::Mat temp_act;
        in["activation_l" + std::to_string(i)] >> temp_act;
        bool all_same = true;
        cv::Mat activation = eigen::eigen2cv_ref(eactivation_);
        ocv::for_each_channels<double>(activation, temp_act,
                                       [&](double lhs, double rhs)
        {
            if(std::abs(lhs - rhs) > 1e-3){
                all_same = false;
            }
        });
        if(!all_same){
            break;
        }else{
            std::cout<<"layer "<<i<<" pass\n";
        }
    }//*/
#endif
}

void autoencoder::update_weight_and_bias(layer &ls)
{
    update_weight_and_bias(ls.w1_grad_, ls.w1_);
    update_weight_and_bias(ls.w2_grad_, ls.w2_);
    update_weight_and_bias(ls.b1_grad_, ls.b1_);
    update_weight_and_bias(ls.b2_grad_, ls.b2_);
}

void autoencoder::
update_weight_and_bias(EigenMat const &bias,
                       EigenMat &weight)
{        
    weight = weight.array() -
            params_.lrate_ * bias.array();
}

autoencoder::criteria::criteria() :
    batch_size_{100},
    beta_{3},
    eps_{5e-5},
    lambda_{3e-3},
    lrate_{2e-2},
    max_iter_{80000},
    sparse_{0.1}
{

}

autoencoder::cv_layer::cv_layer() :
    cost_{0}
{        
}

autoencoder::cv_layer::
cv_layer(int input_size, int hidden_size,
         int mat_type, double cost) :
    cost_{cost}
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
}

void autoencoder::buffer::clear()
{
    delta2_.resize(0, 0);
    delta3_.resize(0, 0);
    delta_buffer_.resize(0, 0);
    pj_.resize(0, 0);
    pj_r0_.resize(0, 0);
    pj_r1_.resize(0, 0);
}

void autoencoder::activation::clear()
{
    hidden_.resize(0, 0);
    output_.resize(0, 0);
}

autoencoder::layer::layer() :
    cost_{0.0}
{

}

autoencoder::layer::
layer(int input_size, int hidden_size,
      double cost) :
    cost_{cost}
{
    w1_ = EigenMat::Random(hidden_size, input_size);
    w2_ = EigenMat::Random(input_size, hidden_size);
    b1_ = EigenMat::Random(hidden_size, 1);
    b2_ = EigenMat::Random(input_size, 1);

    w1_grad_ = EigenMat::Zero(hidden_size, input_size);
    w2_grad_ = EigenMat::Zero(input_size, hidden_size);
    b1_grad_ = EigenMat::Zero(hidden_size, 1);
    b2_grad_ = EigenMat::Zero(input_size, 1);
}

}}

