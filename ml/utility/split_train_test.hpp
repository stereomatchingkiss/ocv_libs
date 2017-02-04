#ifndef OCV_ML_SPLIT_TRAIN_TEST_HPP
#define OCV_ML_SPLIT_TRAIN_TEST_HPP

/*! \file split_train_test.hpp
    \brief split input data to two sets of data
*/

#include <map>
#include <random>
#include <set>
#include <type_traits>
#include <tuple>
#include <vector>

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

/**
 * split input_data and input_label to k set data.
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @return std::map<label_type, std::vector<std::vector<data_type>>.
 * Key is the label, value is the data sets of each label.
 * @code
 * std::vector<std::string> input_data;
 * std::vector<std::string> input_label;
 * std::tie(input_data, input_label) = read_data();
 * //split input_data and input_label into 5 sets of data
 * auto result = split_to_k_set_balance(input_data, input_label, 5);
 * //get instance of label 0
 * std::vector<std::vector<int>> const &instance = result[0];
 * //After split, the result will split the data of every labels into 5 sets.
 * //If label 0 got 10000, after split to 5 sets, every set will contain
 * //2000 instance.
 * for(auto const &vec : instance){
 *   //will print 2000 if original label 0 got 10000 instance
 *   std::cout<<vec.size()<<std::endl;
 * }
 * @endcode
 */
template<typename Data, typename Label>
auto split_to_kset_balance(Data &input_data, Label &input_label, size_t kset)
{
    static_assert((std::is_copy_constructible<Data>::value &&
                   std::is_copy_constructible<Label>::value) ||
                  (std::is_move_constructible<Data>::value &&
                   std::is_move_constructible<Label>::value),
                  "type Data and type Label must be copy constructible or "
                  "move constructible");

    using label_type = typename std::decay<decltype(input_label[0])>::type;
    using data_type = typename std::decay<decltype(input_data[0])>::type;

    if(input_data.size() != input_label.size()){
        throw std::runtime_error("runtime error of " + std::string(__func__) +
                                 " : input_data.size() != input_label.size()");
    }

    std::map<label_type, std::vector<data_type>> category;
    for(size_t i = 0; i != input_data.size(); ++i){
        auto it = category.find(input_label[i]);
        if(it != std::end(category)){
            (it->second).emplace_back(std::move(input_data[i]));
        }else{
            category.insert({input_label[i], {}});
        }
    }

    std::map<label_type, std::vector<std::vector<data_type>>> results;
    for(auto &pair : category){
        size_t const set_size = pair.second.size() / kset;
        size_t const residual = pair.second.size() % kset;
        std::vector<std::vector<data_type>> data_sets;
        auto &pdata = pair.second;
        for(size_t i = 0; i != kset; ++i){
            std::vector<data_type> data;
            size_t const cur_size = i != kset - 1 ? set_size : set_size + residual;
            for(size_t j = 0; j != cur_size && !pdata.empty(); ++j){
                data.emplace_back(std::move(pdata.back()));
                pdata.pop_back();
            }
            data_sets.emplace_back(std::move(data));
        }
        results.insert({pair.first, std::move(data_sets)});
    }

    return results;
}


/**
 * split input data and input label to two sets of data, this function
 * may change the content of input_data and input_label if they can be
 * swapped, this way it could avoid expensive copy operation. Difference
 * part with "split_train_test_inplace" is this function will guarantee
 * the data being split are balanced.
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @param train_data training data split from input data
 * @param train_label training label split from input data
 * @param test_data test data split from input data
 * @param test_label test label split from input data
 * @param test_ratio determine the ratio of the input data
 * split to test data
 * @param shuffle if true, shuffle the train data and label;
 * else do not shuffle.Default value is true
 * @code
 * vector<string> input_list;
 * vector<string> input_label;
 * //read 1000 dogs and 1000 cats
 * std::tie(input_list, input_label) = read_data();
 * vector<string> train_list;
 * vector<string> train_label;
 * vector<string> test_list;
 * vector<string> test_label;
 * //after split, it will make sure train_list got 75% dog and cat,
 * //test list got 25% dog and cat
 * split_train_test_inplace_balance(input_list, input_label,
 * train_list, train_label, test_list, test_label, 0.25);
 * @endcode
 */
template<typename Data, typename Label>
void split_train_test_inplace_balance(Data &input_data, Label &input_label,
                                      Data &train_data, Label &train_label,
                                      Data &test_data, Label &test_label,
                                      double test_ratio)
{
    static_assert((std::is_copy_constructible<Data>::value &&
                   std::is_copy_constructible<Label>::value) ||
                  (std::is_move_constructible<Data>::value &&
                   std::is_move_constructible<Label>::value),
                  "type Data and type Label must be copy constructible or "
                  "move constructible");

    using label_type = typename std::decay<decltype(input_label[0])>::type;
    using data_type = typename std::decay<decltype(input_data[0])>::type;

    std::map<label_type, std::vector<data_type>> category;
    for(size_t i = 0; i != input_data.size(); ++i){
        auto it = category.find(input_label[i]);
        if(it != std::end(category)){
            (it->second).emplace_back(std::move(input_data[i]));
        }else{
            category.insert({input_label[i], {}});
        }
    }

    for(auto &pair : category){
        size_t const test_size =
                static_cast<size_t>(pair.second.size() * test_ratio);
        for(size_t i = 0; i != test_size; ++i){
            test_data.emplace_back(pair.second[i]);
            test_label.emplace_back(pair.first);
        }
        for(size_t i = test_size; i != pair.second.size(); ++i){
            train_data.emplace_back(pair.second[i]);
            train_label.emplace_back(pair.first);
        }
    }
}

template<typename Data, typename Label>
std::tuple<Data, Label, Data, Label>
split_train_test_inplace_balance(Data &input_data, Label &input_label,
                                 double test_ratio)
{
    Data test_data;
    Label test_label;
    Data train_data;
    Label train_label;

    split_train_test_inplace_balance(input_data, input_label, train_data, train_label,
                                     test_data, test_label, test_ratio);

    return std::make_tuple(train_data, train_label,
                           test_data, test_label);
}

/**
 * split input data to two sets of data, this function
 * may clear the content of input can be moved.
 * @param input_data input data want to split
 * @param train_data training data split from input data
 * @param test_data test data split from input data
 * @param test_ratio determine the ratio of the input data
 * split to test data
 */
template<typename Data>
void split_train_test_inplace(Data &input, Data &train_data, Data &test_data, double test_ratio)
{
    size_t const test_size = input.size() * test_ratio;
    std::move(std::begin(input), std::begin(input) + test_size,
              std::back_inserter(test_data));
    std::move(std::begin(input) + test_size, std::end(input),
              std::back_inserter(train_data));
}

/**
 * split input data and input label to two sets of data, this function
 * may change the content of input_data and input_label if they can be
 * swapped, this way it could avoid expensive copy operation
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @param train_data training data split from input data
 * @param train_label training label split from input data
 * @param test_data test data split from input data
 * @param test_label test label split from input data
 * @param test_ratio determine the ratio of the input data
 * split to test data
 * @param shuffle if true, shuffle the train data and label;
 * else do not shuffle.Default value is true
 */
template<typename Data, typename Label>
void
split_train_test_inplace(Data &input_data, Label &input_label,
                         Data &train_data, Label &train_label,
                         Data &test_data, Label &test_label,
                         double test_ratio,
                         bool shuffle = true)
{
    static_assert((std::is_copy_constructible<Data>::value &&
                   std::is_copy_constructible<Label>::value) ||
                  (std::is_move_constructible<Data>::value &&
                   std::is_move_constructible<Label>::value),
                  "type Data and type Label must be copy constructible or "
                  "move constructible");

    size_t const train_size = input_data.size() -
            static_cast<size_t>(input_data.size() * test_ratio);
    enum class tag : unsigned char{
        test_tag,
        train_tag,
    };
    std::vector<tag> seed(input_data.size(), tag::train_tag);
    for(size_t i = train_size; i != input_data.size(); ++i){
        seed[i] = tag::test_tag;
    }
    if(shuffle){
        std::random_device rd;
        std::default_random_engine g(rd());
        std::shuffle(std::begin(seed), std::end(seed), g);
    }

    size_t const test_size = input_data.size() - train_size;
    test_data.resize(test_size);
    test_label.resize(test_size);
    train_data.resize(train_size);
    train_label.resize(train_size);
    size_t test_index = 0;
    size_t train_index = 0;
    for(size_t i = 0; i != seed.size(); ++i){
        if(seed[i] == tag::train_tag){
            std::swap(train_label[train_index], input_label[i]);
            std::swap(train_data[train_index], input_data[i]);
            ++train_index;
        }else{
            std::swap(test_label[test_index], input_label[i]);
            std::swap(test_data[test_index], input_data[i]);
            ++test_index;
        }
    }
}

/**
 * split input data and input label to two sets of data, this function
 * may change the content of input_data and input_label if they can be
 * swapped, this way it could avoid expensive copy operation
 * @param input_data input data want to split
 * @param input_label input label want to split
 * @param test_ratio determine the ratio of the input data
 * split to test data
 * @return train_data, train_label, test_data, test_label
 * @param shuffle if true, shuffle the train data and label;
 * else do not shuffle.Default value is true
 * @warning cannot compile if type Data and Label are not copy constructible
 * or move constructible
 */
template<typename Data, typename Label>
std::tuple<Data, Label, Data, Label>
split_train_test_inplace(Data &input_data, Label &input_label,
                         double test_ratio,
                         bool shuffle = true)
{    
    Data test_data;
    Label test_label;
    Data train_data;
    Label train_label;

    split_train_test_inplace(input_data, input_label, train_data, train_label,
                             test_data, test_label, test_ratio, shuffle);

    return std::make_tuple(train_data, train_label,
                           test_data, test_label);
}

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_ML_SPLIT_TRAIN_TEST_HPP

