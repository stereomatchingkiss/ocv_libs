#ifndef OCV_CBIR_SEARCHER_HPP
#define OCV_CBIR_SEARCHER_HPP

#include "../arma/dist_metric.hpp"
#include "../arma/type_traits.hpp"

#include <armadillo>

#include <boost/container/small_vector.hpp>

#include <map>
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

/**
 *search the vocabulary closest to the histogram
 *@tparam InvertIndex inverted index of the data set
 *@tparam DistMetric policy to measure similarity of
 * histograms
 */
template<typename InvertIndex,
         typename DistMetric = armd::chi_square<>,
         typename Compare = std::less<typename DistMetric::result_type>>
class searcher{
public:
    using invert_value_type = typename InvertIndex::value_type;

    using result_type =
    boost::container::small_vector<invert_value_type, 16>;

    /**
     * @param index inverted index of the dataset
     * @param dist_metric distance metric to measure the similarity
     * of histograms
     * @param result_size determine the size of the results
     * @param candidate_size determine the size of the histograms
     * contains most features
     */
    explicit searcher(InvertIndex index,
                      DistMetric dist_metric = DistMetric(),
                      size_t result_size = 16,
                      size_t candidate_size = 200) :
        candidate_size_{candidate_size},
        dist_metric_(std::move(dist_metric)),
        index_(std::move(index)),
        result_size_{result_size}
    {}

    /**
     * Overload of constructor
     */
    explicit searcher(InvertIndex index,
                      size_t result_size = 16,
                      size_t candidate_size = 200) :
        candidate_size_{candidate_size},
        index_(std::move(index)),
        result_size_{result_size}
    {}

    size_t get_candidate_size() const
    {
        return candidate_size_;
    }

    size_t get_result_size() const
    {
        return result_size_;
    }

    void set_candidate_size(size_t value)
    {
        candidate_size_ = value;
    }

    void set_result_size(size_t value)
    {
        result_size_ = value;
    }

    /**
     *search the histogram closest to the query histogram
     *@param query_hist self explain
     *@param dataset_hist self explain
     */
    template<typename QueryHist, typename DataHist>
    result_type search(QueryHist const &query_hist,
                       DataHist const &dataset_hist) const
    {
        return filter_candidate(find_candidate(query_hist),
                                query_hist, dataset_hist);
    }

private:
    using candidate_type = std::map<invert_value_type, size_t>;
    using candidate_value_type =
    std::pair<
    typename std::remove_const<typename candidate_type::key_type>::type,
    typename candidate_type::mapped_type
    >;
    using cvt = candidate_value_type;

    template<typename QueryHist, typename DataHist>
    result_type filter_candidate(std::vector<cvt> const &candidate,
                                 QueryHist const &query_hist,
                                 DataHist const &dataset_hist) const
    {
        static_assert(armd::is_two_dim<DataHist>::value,
                      "DataHist should be arma::Mat or arma::SpMat");

        using namespace boost::container;

        using dist_type = typename DistMetric::result_type;
        using sm_vtype = std::pair<dist_type, invert_value_type>;

        small_vector<sm_vtype, 16> sm;
        for(auto const &val : candidate){
            auto const dist =
                    dist_metric_.compare(query_hist,
                                         dataset_hist,
                                         val.first);//*/
            if(sm.size() < result_size_){
                sm.emplace_back(dist, val.first);
                if(sm.size() == result_size_){
                    auto func = [](sm_vtype const &lhs, sm_vtype const &rhs)
                    {
                        return Compare()(lhs.first,rhs.first);
                    };
                    std::sort(std::begin(sm), std::end(sm),
                              func);
                }
            }else{
                auto it = std::find_if(std::begin(sm), std::end(sm),
                                       [=](sm_vtype const &a)
                {                    
                    return Compare()(dist,a.first);
                });
                if(it != std::end(sm)){
                    sm.insert(it, {dist, val.first});
                    sm.pop_back();
                }
            }
        }

        result_type result;
        for(auto const &val : sm){
            result.push_back(val.second);
        }

        return result;
    }

    template<typename T>
    std::vector<cvt>
    find_candidate(arma::Mat<T> const &query_hist) const
    {
        candidate_type candidate;
        for(arma::uword i = 0; i != query_hist.n_rows; ++i){
            auto inv_it = index_.find(query_hist(i));
            if(inv_it != std::end(index_)){
                auto const &words = inv_it->second;
                for(auto const &val : words){
                    auto m_it = candidate.find(val);
                    if(m_it != std::end(candidate)){
                        ++(m_it->second);
                    }else{
                        candidate.insert({val, 1});
                    }
                }
            }
        }

        return sort_candidate(candidate);
    }

    template<typename QueryHist>
    std::vector<cvt>
    find_candidate(QueryHist const &query_hist) const
    {
        candidate_type candidate;
        for(auto it = std::begin(query_hist);
            it != std::end(query_hist); ++it){
            auto inv_it = index_.find(it.row());
            if(inv_it != std::end(index_)){
                auto const &words = inv_it->second;
                for(auto const &val : words){
                    auto m_it = candidate.find(val);
                    if(m_it != std::end(candidate)){
                        ++(m_it->second);
                    }else{
                        candidate.insert({val, 1});
                    }
                }
            }
        }

        return sort_candidate(candidate);
    }

    std::vector<cvt>
    sort_candidate(candidate_type const &candidate) const
    {
        std::vector<cvt> sorted(candidate.begin(), candidate.end());
        auto sort_criteria = [](cvt const &lhs, cvt const &rhs)
        {
            return lhs.second > rhs.second;
        };
        std::sort(std::begin(sorted),
                  std::end(sorted), sort_criteria);
        sorted.resize(std::min(candidate_size_, candidate.size()));

        return sorted;
    }

    size_t candidate_size_;
    DistMetric dist_metric_;
    InvertIndex index_;
    size_t result_size_;
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_SEARCHER_HPP
