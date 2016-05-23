#ifndef OCV_TINY_CNN_PREDICTOR_HPP
#define OCV_TINY_CNN_PREDICTOR_HPP

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

/**
 * predict the result by tiny_cnn net
 *@code
 *using NetType =
    tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adam>;
 *predictor<NetType> pd;
 *auto const img = cv::imread("so.jpg");
 *auto const results = pd.predict(ocv::tcnn::cvmat_to_img(img));
 *for(auto const &val : results){
 *  std::cout<<"probability : "<<val.first<<
 *  ", label = "<<val.second<<std::endl;
 *}
 *@endcode
 */
template<typename Net>
class predictor
{
public:
    using result_type = std::vector<std::pair<double, int>>;

    explicit predictor(Net net);

    template<typename T>
    result_type predict(T const &input) const
    {
        result_type result;
        predict(input, result);

        return result;
    }

    template<typename T>
    void predict(T const &input, result_type &result) const
    {
      auto const prob = nn_.predict(input);
      for(size_t i = 0; i != prob.size(); ++i){
          result.emplace_back(prob[i], i);
      }
    }

private:
    Net nn_;
};

template<typename Net>
predictor<Net>::predictor(Net net) :
    nn_(std::move(net))
{

}


} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // PREDICTOR_HPP
