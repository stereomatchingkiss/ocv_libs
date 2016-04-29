#ifndef OCV_CBIR_SPATIAL_VERIFIER_HPP
#define OCV_CBIR_SPATIAL_VERIFIER_HPP

#include "../arma/type_traits.hpp"
#include "features_indexer.hpp"

#include <armadillo>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup cbir
 *  @{
 */
namespace cbir{

template<typename CodeBook, typename FeaturesType,
         int BufSize = 20>
class spatial_verifier
{
public:
    using result_type =
    boost::container::small_vector<size_t, BufSize>;

    explicit
    spatial_verifier(features_indexer const &fi,
                     CodeBook const &code_book,
                     float ratio = 0.75,
                     size_t min_matches = 10,
                     size_t num_result = 16,
                     double reproj_thresh = 4.0) :
        code_book_(code_book),
        fi_(fi),
        min_matches_{min_matches},
        num_result_{num_result},
        ratio_{ratio},
        reproj_thresh_{reproj_thresh}
    {
        static_assert(std::is_arithmetic<FeaturesType>::value,
                      "FeaturesType should be arimetic");
        static_assert(armd::is_two_dim<CodeBook>::value,
                      "CodeBook should be arma::Mat or arma::SpMat");
    }

    size_t get_min_matches() const
    {
        return min_matches_;
    }

    size_t get_num_result() const
    {
        return num_result_;
    }

    double get_ratio() const
    {
        return ratio_;
    }

    double get_reproj_thresh() const
    {
        return reproj_thresh_;
    }


    template<typename T>
    result_type
    rerank(std::vector<cv::KeyPoint> const &query_kp,
           cv::Mat const &query_features,
           T &&search_results) const
    {
        using decay_type = typename std::decay<T>::type;
        using value_type = typename decay_type::value_type;
        using rerank_type =
        boost::container::small_vector<std::pair<size_t, value_type>, 16>;
        using rerank_vtype = typename rerank_type::value_type;

        static_assert(std::is_class<decay_type>::value,
                      "T should be class or struct");
        static_assert(std::is_copy_assignable<decay_type>::value ||
                      std::is_move_assignable<T>::value,
                      "T should be copy assignable");
        static_assert(std::is_integral<value_type>::value,
                      "The value_type of T should be intergral");

        decay_type sort_results = std::forward<T>(search_results);
        std::sort(std::begin(sort_results), std::end(sort_results));

        std::vector<cv::KeyPoint> kp;
        cv::Mat features;
        rerank_type rerank;
        rerank.reserve(sort_results.size());
        for(auto const val : sort_results){
            fi_.read_keypoints(kp, val);
            fi_.read_image_features(features, val);
            auto const match_pt_size =
                    match(query_kp, query_features,
                          kp, features);
            rerank.emplace_back(match_pt_size, val);
        }

        std::sort(std::begin(rerank), std::end(rerank),
                  [](rerank_vtype const &lhs, rerank_vtype const &rhs)
        {
            return lhs.first > rhs.first;
        });
        result_type result;
        result.reserve(rerank.size());
        for(auto const &val : rerank){
            result.emplace_back(val.second);
        }

        return result;
    }

    void set_min_matches(size_t value)
    {
        min_matches_ = value;
    }

    void set_num_result(size_t value)
    {
        num_result_ = value;
    }

    void set_ratio(double value)
    {
        ratio_ = value;
    }

    void set_reproj_thresh(double value)
    {
        reproj_thresh_ = value;
    }

private:
    size_t match(std::vector<cv::KeyPoint> const &query_kp,
                 cv::Mat const &query_features,
                 std::vector<cv::KeyPoint> const &target_kp,
                 cv::Mat const &target_features) const
    {
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> nn_matches;
        matcher.knnMatch(query_features, target_features,
                         nn_matches, 2);

        std::vector<cv::Point2f> matched1, matched2;
        for(size_t i = 0; i < nn_matches.size(); i++){
            if(nn_matches[i].size() == 2){
                float const dist1 = nn_matches[i][0].distance;
                float const dist2 = nn_matches[i][1].distance;

                if(dist1 < ratio_ * dist2) {
                    cv::DMatch const &first = nn_matches[i][0];
                    matched1.push_back(query_kp[first.queryIdx].pt);
                    matched2.push_back(target_kp[first.trainIdx].pt);
                }
            }
        }

        if(matched1.size() >= min_matches_){
            auto const homo =
                    cv::findHomography(matched1, matched2,
                                       cv::RANSAC,
                                       reproj_thresh_);
            if(!homo.empty()){
                size_t sum = 0;
                for(size_t i = 0; i != matched1.size(); ++i){
                    cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
                    col.at<double>(0) = matched1[i].x;
                    col.at<double>(1) = matched1[i].y;
                    col = homo * col;
                    col /= (col.at<double>(2) + 1e-10);
                    double const dist =
                            std::sqrt(std::pow(col.at<double>(0)
                                               - matched2[i].x, 2) +
                                      std::pow(col.at<double>(1) -
                                               matched2[i].y, 2));
                    if(dist < reproj_thresh_) {
                        ++sum;
                    }
                }
                return sum;
            }
        }

        return 0;
    }

    CodeBook const &code_book_;
    features_indexer const &fi_;
    size_t min_matches_;
    size_t num_result_;
    float ratio_;
    double reproj_thresh_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_SPATIAL_VERIFIER_HPP
