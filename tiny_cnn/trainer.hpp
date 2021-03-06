#ifndef OCV_TRAINER_HPP
#define OCV_TRAINER_HPP

#include <boost/progress.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup tcnn
 *  @{
 */
namespace tcnn{

class trainer
{
public:
    explicit trainer(std::string output_file) :
        output_file_(std::move(output_file))
    {}

    trainer(std::string output_file,
            size_t minibatch_size, int num_epochs) :
        minibatch_size_{minibatch_size},
        num_epochs_{num_epochs},
        output_file_(std::move(output_file))
    {}

    size_t get_minibatch_size() const
    {
        return minibatch_size_;
    }

    int get_num_epoch() const
    {
        return num_epochs_;
    }

    std::string const &get_output_file() const
    {
        return output_file_;
    }

    void set_minibatch_size(size_t value)
    {
        minibatch_size_ = value;
    }

    void set_num_epoch(int value)
    {
        num_epochs_ = value;
    }

    void set_output_file(std::string const &output_file)
    {
        output_file_ = output_file;
    }

    template<typename Net, typename Img, typename Label>
    void train(Net &net, std::vector<Img> const &train_img,
               std::vector<Label> const &train_label);

    template<typename Net, typename Img, typename Label>
    void train_and_test(Net &net, std::vector<Img> const &train_img,
                        std::vector<Label> const &train_label,
                        std::vector<Img> const &test_img,
                        std::vector<Label> const &test_label,
                        bool reset_weights = true);

    template<typename Net, typename Img, typename Label>
    void train_and_test(Net &net, std::vector<Img> const &train_img,
                        std::vector<Label> const &train_label,
                        std::vector<Img> const &test_img,
                        std::vector<Label> const &test_label,
                        std::ostream &out,
                        bool reset_weights = true);

private:
    size_t minibatch_size_ = 1;
    int num_epochs_ = 30;
    std::string output_file_;
};

template<typename Net, typename Img, typename Label>
void trainer::train_and_test(Net &net, std::vector<Img> const &train_img,
                             std::vector<Label> const &train_label,
                             std::vector<Img> const &test_img,
                             std::vector<Label> const &test_label,
                             std::ostream &out,
                             bool reset_weights)
{
    std::cout << "start learning" << std::endl;
    boost::progress_display disp(static_cast<int>(train_img.size()));
    boost::timer t;
    size_t epoch = 0;
    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout<<"epoch "<<epoch++<<std::endl;
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        auto res = net.test(test_img, test_label);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(static_cast<int>(train_img.size()));
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += static_cast<unsigned long>(minibatch_size_);
    };

    // training
    net.train(train_img, train_label, minibatch_size_, num_epochs_,
              on_enumerate_minibatch, on_enumerate_epoch,
              reset_weights);

    std::cout << "end training." << std::endl;

    // save networks
    std::ofstream ofs(output_file_);
    ofs << net;
    // test and show results
    net.test(test_img, test_label).print_detail(out);
}

template<typename Net, typename Img, typename Label>
inline
void trainer::train_and_test(Net &net, std::vector<Img> const &train_img,
                             std::vector<Label> const &train_label,
                             std::vector<Img> const &test_img,
                             std::vector<Label> const &test_label,
                             bool reset_weights)
{
    train_and_test(net, train_img, train_label,
                   test_img, test_label, std::cout,
                   reset_weights);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // TRAINER_HPP
