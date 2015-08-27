#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../eigen/eigen.hpp"

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include <random>
#include <vector>

/*! \file softmax.hpp
    \brief implement the algorithm--softmax regression based on\n
    the description of UFLDL, these codes are develop based\n
    on the example on the website(http://eric-yuan.me/softmax-regression-cv/#comment-8781).
    By now this class only support double as input data
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

class softmax
{
public:
    using EigenMat = eigen::MatRowMajor<double>;

    softmax();

    EigenMat const& get_weight() const;

    void set_batch_size(int batch_size);
    void set_lambda(double lambda);
    void set_learning_rate(double lrate);
    void set_max_iter(size_t max_iter);

    void train(EigenMat const &train,
               std::vector<int> const &labels);

private:
    double compute_cost(EigenMat const &train,
                        EigenMat const &weight);

    void compute_hypothesis(EigenMat const &train,
                           EigenMat const &weight);
    void compute_gradient(EigenMat const &train);

    struct params
    {
        params();
        int batch_size_;
        double cost_;
        double inaccuracy_;
        double lambda_;
        double lrate_;
        size_t max_iter_;
    };

    EigenMat hypothesis_;
    EigenMat grad_;
    EigenMat ground_truth_;
    EigenMat max_exp_power_;
    params params_;
    EigenMat weight_;
    EigenMat weight_sum_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // SOFTMAX_H
