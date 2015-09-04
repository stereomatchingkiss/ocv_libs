#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "../../core/utility.hpp"
#include "../../eigen/eigen.hpp"

#include "../../profile/measure.hpp"
#include "propagation.hpp"

#include <Eigen/Dense>

#include <dlib/optimization.h>

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

        size_t size_of_weights() const
        {
            return w1_.size() + w2_.size() + b1_.size() + b2_.size();
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

    EigenMat const& get_activation() const
    {
        return eactivation_;
    }
    std::vector<layer> const& get_layer() const
    {
        return layers_;
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

        for(size_t i = 0; i < params_.hidden_size_.size(); ++i){
            Eigen::MatrixBase<Derived> const &TmpInput =
                    i == 0 ? input
                           : eactivation_;
            if(!reuse_layer_){
                layer es(TmpInput.rows(), params_.hidden_size_[i]);
                reduce_cost(TmpInput, es);
                generate_activation(es, TmpInput,
                                    i==0?true:false);
                layers_.push_back(es);
            }else{
                reduce_cost(TmpInput, layers_[i]);
                generate_activation(layers_[i], TmpInput,
                                    i==0?true:false);
            }
        }
        act_.clear();
        buffer_.clear();//*/
    }


private:
    using MatType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mapper = Eigen::Map<MatType, Eigen::Aligned>;    
    using ColVec = dlib::matrix<double, 0, 1>;

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

    template<typename Derived>
    size_t convert(Eigen::MatrixBase<Derived> const &input,
                   ColVec &output, size_t begin) const
    {
        for(size_t row = 0; row != input.rows(); ++row){
            for(size_t col = 0; col != input.cols(); ++col){
                output(begin++) = input(row, col);
            }
        }
        return begin;
    }

    template<typename Derived>
    size_t convert(ColVec const &input,
                   Eigen::MatrixBase<Derived> &output,
                   size_t begin) const
    {
        for(size_t row = 0; row != output.rows(); ++row){
            for(size_t col = 0; col != output.cols(); ++col){
                output(row, col) = input(begin++);
            }
        }
        return begin;
    }

    void convert_grad(layer const &input,
                      ColVec &output) const
    {
        size_t index = convert(input.w1_grad_, output, 0);
        index = convert(input.w2_grad_, output, index);
        index = convert(input.b1_grad_, output, index);
        convert(input.b2_grad_, output, index);
    }

    void convert_weights(layer const &input,
                         ColVec &output) const
    {
        size_t index = convert(input.w1_, output, 0);
        index = convert(input.w2_, output, index);
        index = convert(input.b1_, output, index);
        convert(input.b2_, output, index);
    }

    void convert_weights(ColVec const &input,
                         layer &output) const
    {
        size_t index = convert(input, output.w1_, 0);
        index = convert(input, output.w2_, index);
        index = convert(input, output.b1_, index);
        convert(input, output.b2_, index);
    }    

    template<typename Derived>
    void compute_cost(Eigen::MatrixBase<Derived> const &input,
                      layer &es)
    {
        //std::cout<<&input(0, 0)<<"\n";
        get_activation(input, es);
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

        forward_propagation(temp_input, ls.w1_,
                            ls.b1_, eactivation_,
                            no_overlap);
    }    

    template<typename Derived>
    void get_activation(Eigen::MatrixBase<Derived> const &input,
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

    template<typename Derived>
    void reduce_cost(Eigen::MatrixBase<Derived> const &input,
                     layer &ls)
    {
        ColVec cvec(ls.size_of_weights());
        convert_weights(ls, cvec);

        auto cost_func = [&](ColVec const &vec)->double
        {
            convert_weights(vec, ls);
            compute_cost(input, ls);

            return ls.cost_;
        };

        auto grad_func = [&](ColVec const&)->ColVec
        {
            compute_gradient(input, ls);
            ColVec result(ls.size_of_weights());
            convert_grad(ls, result);

            return result;
        };

        dlib::find_min(dlib::lbfgs_search_strategy(10),
                       dlib::objective_delta_stop_strategy(params_.eps_).
                       be_verbose(),
                       cost_func, grad_func, cvec, 0);
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
