#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "../../core/utility.hpp"
#include "../../eigen/eigen.hpp"

#ifdef OCV_TEST_AUTOENCODER
#include "../utility/gradient_checking.hpp"
#endif

#include "../../profile/measure.hpp"
#include "propagation.hpp"

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

#include <fstream>
#include <random>

/*! \file autoencoder.hpp
    \brief implement the algorithm--autoencoder based on\n
    the description of UFLDL, these codes are develop based\n
    on the example on the website(http://eric-yuan.me/simple-deep-network/)
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
class autoencoder
{
public:
    using EigenMat = eigen::MatRowMajor<T>;

    struct layer
    {
        layer() :
            cost_{0.0}
        {

        }
        layer(int input_size, int hidden_size,
              double cost = 0) :
            cost_{cost}
        {            
            std::random_device rd;
            std::default_random_engine re(rd());
            std::uniform_real_distribution<T> ur(0, 1);
            ur(re);
            w1_.resize(hidden_size, input_size);
            w2_.resize(input_size, hidden_size);
            b1_ = EigenMat::Zero(hidden_size, 1);
            b2_ = EigenMat::Zero(input_size, 1);

            for(size_t row = 0; row != w1_.rows(); ++row){
                for(size_t col = 0; col != w1_.cols(); ++col){
                    w1_(row, col) = ur(re);
                }
            }
            for(size_t row = 0; row != w2_.rows(); ++row){
                for(size_t col = 0; col != w2_.cols(); ++col){
                    w2_(row, col) = ur(re);
                }
            }

            double const R  = std::sqrt(6) /
                    std::sqrt(hidden_size + input_size + 1);
            w1_ = w1_.array() * 2 * R - R;
            w2_ = w2_.array() * 2 * R - R;

            w1_grad_ = EigenMat::Zero(hidden_size, input_size);
            w2_grad_ = EigenMat::Zero(input_size, hidden_size);
            b1_grad_ = EigenMat::Zero(hidden_size, 1);
            b2_grad_ = EigenMat::Zero(input_size, 1);
        }

        EigenMat w1_;
        EigenMat w2_;
        EigenMat b1_;
        EigenMat b2_;
        EigenMat w1_grad_;
        EigenMat w2_grad_;
        EigenMat b1_grad_;
        EigenMat b2_grad_;
        double cost_;
    };

    explicit autoencoder(cv::AutoBuffer<int> const &hidden_size) :
        reuse_layer_{false}
    {
        set_hidden_layer_size(hidden_size);
    }

    autoencoder& operator=(autoencoder const&) = delete;
    autoencoder& operator=(autoencoder &&) = delete;
    autoencoder(autoencoder const&) = delete;
    autoencoder(autoencoder &&) = delete;

    void clear_decode_result()
    {
        eactivation_.resize(0, 0);
    }

    EigenMat const& get_decode_result() const
    {
        return eactivation_;
    }

    std::vector<layer> const& get_layer() const
    {
        return layers_;
    }

    /**
     * @brief read the train result from the file(xml)
     * @param file the file save the train result
     */
    void read(std::string const &file)
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
    void set_batch_size(int size)
    {
        params_.batch_size_ = size;
    }

    /**
     * @brief set the beta of autoencoder
     * @param beta the weight of the sparsity penalty term.\n
     * Must be real positive number.The default value is 3
     */
    void set_beta(double beta)
    {
        params_.beta_ = beta;
    }

    /**
         * @brief set the epsillon
         * @param eps The desired accuracy or change in parameters\n
         *  at which the iterative algorithm stops.
         */
    void set_epsillon(double eps){
        params_.eps_ = eps;
    }

    /**
     * @brief set the hidden layers size
     * @param size size of each hidden layers,must bigger\n
     * than zero.The default value is 0
     */
    void set_hidden_layer_size(cv::AutoBuffer<int> const &size)
    {
        params_.hidden_size_ = size;
    }

    /**
     * @brief set the lambda of the regularization term
     * @param lambda the weight of the regularization term.\n
     * Must be real positive number.The default value is 3e-3
     */
    void set_lambda(double lambda)
    {
        params_.lambda_ = lambda;
    }

    /**
     * @brief set the learning rate
     * @param The larger the lrate, the faster we approach the solution,\n
     *  but larger lrate may incurr divergence, must be real\n
     *  positive number.The default value is 2e-2
     */
    void set_learning_rate(double lrate)
    {
        params_.lrate_ = lrate;
    }

    /**
     * @brief set maximum iteration time
     * @param iter the maximum iteration time of the algorithm.\n
     * The default value is 80000
     */
    void set_max_iter(int iter)
    {

        params_.max_iter_ = iter;
    }

    /**
     * @brief set the sparsity penalty
     * @param sparse Constraint of the hidden neuron, the lower it is,\n
     * the sparser the output of the layer would be.The default\n
     * value is 0.1
     */
    void set_sparse(double sparse)
    {
        params_.sparse_ = sparse;
    }

    /**
     * @brief set the trained layer should be reuse or not
     * @param reuse true will reuse the trained layer if exist;else\n
     * the train function will start a new training process
     */
    void set_reuse_layer(bool reuse)
    {
        reuse_layer_ = reuse;
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
    template<typename Derived>
    void train(Eigen::MatrixBase<Derived> const &input)
    {
        if(!reuse_layer_){
            layers_.clear();
        }
        std::random_device rd;
        std::default_random_engine re(rd());
        int const Batch = get_batch_size(input.cols());
        int const RandomSize = input.cols() != Batch ?
                    input.cols() - Batch - 1 : 0;
        std::uniform_int_distribution<int>
                uni_int(0, RandomSize);

#ifdef OCV_TEST_AUTOENCODER
        gradient_check();
#endif

        for(size_t i = 0; i < params_.hidden_size_.size(); ++i){
            Eigen::MatrixBase<Derived> const &TmpInput =
                    i == 0 ? input
                           : eactivation_;
            if(!reuse_layer_){
                layer es(TmpInput.rows(), params_.hidden_size_[i]);               
                reduce_cost(uni_int, re, Batch, TmpInput, es);
                generate_activation(es, TmpInput,
                                    i==0?true:false);
                layers_.push_back(es);
            }else{
                reduce_cost(uni_int, re, Batch, TmpInput, layers_[i]);
                generate_activation(layers_[i], TmpInput,
                                    i==0?true:false);
            }
        }
        act_.clear();
        buffer_.clear();//*/
    }

    /**
     * @brief write the training result into the file(xml)
     * @param file the name of the file
     */
    void write(std::string const &file) const
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
private:
    using MatType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mapper = Eigen::Map<MatType, Eigen::Aligned>;
    using MapperConst = Eigen::Map<const MatType, Eigen::Aligned>;

    struct activation
    {
        void clear()
        {
            hidden_.resize(0, 0);
            output_.resize(0, 0);
        }

        EigenMat hidden_;
        EigenMat output_;
    };

    struct buffer
    {
        void clear()
        {
            delta2_.resize(0, 0);
            delta3_.resize(0, 0);
            delta_buffer_.resize(0, 0);
            pj_.resize(0, 0);
            pj_r0_.resize(0, 0);
            pj_r1_.resize(0, 0);
        }

        EigenMat delta2_;
        EigenMat delta3_;
        EigenMat delta_buffer_;
        EigenMat pj_; //the average activation of hidden units
        EigenMat pj_r0_; //same as pj_ expect 0(set to max() of double)
        EigenMat pj_r1_; //same as pj_ expect 1(set to max() of double)
    };

    struct cv_layer
    {
        cv_layer() :
            cost_{0}
        {
        }
        cv_layer(int input_size, int hidden_size,
                 int mat_type,
                 double cost = 0) :
            cost_{cost}
        {
            w1_.create(hidden_size, input_size, mat_type);
            w2_.create(input_size, hidden_size, mat_type);
            b1_.create(hidden_size, 1, mat_type);
            b2_.create(input_size, 1, mat_type);

            generate_random_value<T>(w1_, 0.12);
            generate_random_value<T>(w2_, 0.12);
            generate_random_value<T>(b1_, 0.12);
            generate_random_value<T>(b2_, 0.12);

            w1_grad_ = cv::Mat::zeros(hidden_size, input_size, mat_type);
            w2_grad_ = cv::Mat::zeros(input_size, hidden_size, mat_type);
            b1_grad_ = cv::Mat::zeros(hidden_size, 1, mat_type);
            b2_grad_ = cv::Mat::zeros(input_size, 1, mat_type);
        }

        cv::Mat w1_;
        cv::Mat w2_;
        cv::Mat b1_;
        cv::Mat b2_;
        cv::Mat w1_grad_;
        cv::Mat w2_grad_;
        cv::Mat b1_grad_;
        cv::Mat b2_grad_;
        double cost_;
    };

    struct criteria
    {
        criteria() :
            batch_size_{100},
            beta_{3},
            eps_{1e-8},
            lambda_{3e-3},
            lrate_{2e-2},
            max_iter_{80000},
            sparse_{0.1}
        {

        }

        int batch_size_;
        double beta_;
        double eps_;
        cv::AutoBuffer<int> hidden_size_;
        double lambda_;
        double lrate_; //learning rate
        int max_iter_;
        double sparse_;
    };

    void convert(cv_layer const &input,
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
    void convert(layer const &input,
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

    template<typename Derived>
    void compute_cost(Eigen::MatrixBase<Derived> const &input,
                      layer &es)
    {
        //std::cout<<&input(0, 0)<<"\n";
        get_decode_result(input, es);
        //std::cout<<"get activation\n";
        auto const NSamples = input.cols();
        //square error of back propagation(first half)
        double const SquareError =
                ((act_.output_.array() - input.array()).pow(2.0) / 2.0).
                sum() / NSamples;
        //std::cout<<"square error : "<<SquareError<<"\n";
        // the second part is weight decay part
        double const WeightError =
                ((es.w1_.array() * es.w1_.array()).sum() +
                 (es.w2_.array() * es.w2_.array()).sum()) *
                (params_.lambda_ / 2.0);
        //std::cout<<"weight error : "<<WeightError<<"\n";

        // now calculate pj which is the average activation of hidden units
        buffer_.pj_ = act_.hidden_.rowwise().sum() / NSamples;
        //prevent division by zero
        buffer_.pj_r0_ = (buffer_.pj_.array() != 0.0).
                select(buffer_.pj_, 1000);
        buffer_.pj_r1_ = (buffer_.pj_.array() != 1.0).
                select(buffer_.pj_, -1000);//*/

        //the third part of overall cost function is the sparsity part
        double const Sparse = params_.sparse_;
        //beta * sum(sparse * log[sparse/pj] +
        //           (1 - sparse) * log[(1-sparse)/(1-pj)])
        double const SparseError =
                ( (Sparse * (Sparse / (buffer_.pj_r0_.array())).log()) +
                  (1.0-Sparse)*((1.0-Sparse)/(1.0-buffer_.pj_r1_.array())).log()).sum() *
                params_.beta_;
        //std::cout<<"Sparse error : "<<SparseError<<"\n";
        es.cost_ = SquareError + WeightError + SparseError;//*/
    }

    template<typename Derived>
    void compute_gradient(Eigen::MatrixBase<Derived> const &input,
                          layer &es)
    {
        auto const NSamples = input.cols();
        buffer_.delta3_ =
                ((act_.output_.array() - input.array()) / NSamples) *
                ((1.0 - act_.output_.array()) * act_.output_.array());
        //std::cout<<buffer_.delta3_<<"\n\n";
        es.w2_grad_.noalias() = buffer_.delta3_*act_.hidden_.transpose();
        es.w2_grad_ = (es.w2_grad_.array()) +
                params_.lambda_ * es.w2_.array();
        es.b2_grad_ = buffer_.delta3_.rowwise().sum();

        get_delta_2(buffer_.delta3_, es, NSamples);
        es.w1_grad_.noalias() = buffer_.delta2_*input.transpose();
        es.w1_grad_ = (es.w1_grad_.array()) +
                params_.lambda_ * es.w1_.array();
        es.b1_grad_ = buffer_.delta2_.rowwise().sum();
    }

    template<typename Derived>
    void generate_activation(layer const &ls,
                             Eigen::MatrixBase<Derived> const &temp_input,
                             bool no_overlap = true)
    {
#ifdef OCV_MEASURE_TIME
        auto const TGen =
                time::measure<>::duration([&]()
        { generate_activation_impl(ls, temp_input,
                                   no_overlap); });
        std::cout<<"time of generate last layer activation : "<<TGen.count()<<"\n\n";
#else
        generate_activation_impl(ls, temp_input, no_overlap);
#endif
    }

    template<typename Derived>
    void generate_activation_impl(layer const &ls,
                                  Eigen::MatrixBase<Derived> const &temp_input,
                                  bool no_overlap = true)
    {
        forward_propagation(temp_input, ls.w1_,
                            ls.b1_, eactivation_,
                            no_overlap);
    }

    template<typename Derived>
    void get_decode_result(Eigen::MatrixBase<Derived> const &input,
                        layer &es)
    {
        forward_propagation(input, es.w1_, es.b1_, act_.hidden_);
        forward_propagation(act_.hidden_, es.w2_, es.b2_, act_.output_);
    }
    int get_batch_size(int sample_size) const
    {
        return std::min(params_.batch_size_, sample_size);
    }

    template<typename Derived>
    void get_delta_2(Eigen::MatrixBase<Derived> const &delta_3,
                     layer const &es,
                     size_t sample_size)
    {
        buffer_.delta_buffer_ =
                (params_.beta_ / sample_size) *
                (-params_.sparse_/(buffer_.pj_r0_.array()) +
                 (1.0-params_.sparse_)/(1.0-buffer_.pj_r1_.array()));

        Mapper Map(buffer_.delta_buffer_.data(),
                   buffer_.delta_buffer_.size());
        buffer_.delta2_.noalias() = es.w2_.transpose() * delta_3;
        buffer_.delta2_.colwise() += Map;
        buffer_.delta2_ =
                buffer_.delta2_.array() *
                ((1.0 - act_.hidden_.array()) * act_.hidden_.array());
    }

    void gradient_check()
    {
        EigenMat const Input = EigenMat::Random(8, 2);
        layer es(Input.rows(), Input.rows() / 2);
        layer es_copy = es;
        gradient_checking gc;
        auto func = [&](EigenMat &theta)->T
        {
            es.w1_.swap(theta);
            compute_cost(Input, es);
            auto const Cost = es.cost_;
            es.w1_.swap(theta);

            return Cost;
        };
        EigenMat const Gradient =
                gc.compute_gradient(es.w1_,
                                    func);

        compute_cost(Input, es_copy);
        compute_gradient(Input, es_copy);

        std::cout<<std::boolalpha<<"pass : "<<
                   gc.compare_gradient(Gradient, es_copy.w1_grad_)<<"\n";
    }

    template<typename Derived>
    void reduce_cost(std::uniform_int_distribution<int> const &uni_int,
                     std::default_random_engine &re,
                     int batch, Eigen::MatrixBase<Derived> const &input,
                     layer &ls)
    {
        double last_cost = 0.0;
        auto const LRate = params_.lrate_;
#ifndef  OCV_MEASURE_TIME
        for(int j = 0; j != params_.max_iter_; ++j){
            int const X = uni_int(re);
            auto const &Temp = input.block(0, X,
                                           input.rows(), batch);
            compute_cost(Temp, ls);            

#ifdef OCV_PRINT_COST
            std::cout<<j<<" : cost : "<<ls.cost_
                    <<", random : "<<X<<"\n";
#endif

            if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                    ls.cost_ <= 0.0){
                break;
            }

            compute_gradient(Temp, ls);
            //std::cout<<ls.w1_<<"\n\n";

            last_cost = ls.cost_;
            update_weight(ls);
            //std::cout<<ls.b2_grad_<<"\n\n";
        }
#else
        double t_cost = 0;
        double t_gra = 0;
        double t_update = 0;
        int iter_time = 1;
        for(int j = 0; j != params_.max_iter_; ++j){
            int const X = uni_int(re);
            auto const &Temp = input.block(0, X,
                                           input.rows(), batch);
            t_cost +=
                    time::measure<>::execution([&]()
            { compute_cost(Temp, ls); });

#ifdef OCV_PRINT_COST
            std::cout<<j<<" : cost : "<<ls.cost_
                    <<", random : "<<X<<"\n";
#endif

            if(std::abs(last_cost - ls.cost_) < params_.eps_ ||
                    ls.cost_ <= 0.0){
                break;
            }

            t_gra +=
                    time::measure<>::execution([&]()
            { compute_gradient(Temp, ls); });
            ++iter_time;

            last_cost = ls.cost_;
            t_update +=
                    time::measure<>::execution([&]()
            { update_weight(ls); });            
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

    void update_weight(layer &ls)
    {
        update_weight(ls.w1_grad_, ls.w1_);
        update_weight(ls.w2_grad_, ls.w2_);
        update_weight(ls.b1_grad_, ls.b1_);
        update_weight(ls.b2_grad_, ls.b2_);
    }

    template<typename Derived>
    void update_weight(Eigen::MatrixBase<Derived> const &gradient,
                       Eigen::MatrixBase<Derived> &weight)
    {
        weight = weight.array() -
                params_.lrate_ * gradient.array();
    }

    activation act_;
    buffer buffer_;
    criteria params_;
    EigenMat eactivation_;
    std::vector<layer> layers_;
    bool reuse_layer_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // AUTOENCODER_H
