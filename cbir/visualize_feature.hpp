#ifndef OCV_CBIR_VISUALIZE_FEATURE_HPP
#define OCV_CBIR_VISUALIZE_FEATURE_HPP

#include "features_indexer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <armadillo>

#include <algorithm>
#include <limits>

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
 *visualize the features of BOVW
 *@tparam T Features type of hdf5
 *@tparam U Type of code book(vocab)
 */
template<typename T, typename U = double>
class visualize_features
{
public:    
    struct vis_point
    {
        vis_point() :
            dist_{std::numeric_limits<double>::max()}
        {}

        vis_point(U dist, std::string img_name,
                  cv::KeyPoint kp) :
            dist_{dist},
            img_name_(std::move(img_name)),
            kp_(kp)
        {}

        friend bool operator<(vis_point const &lhs,
                              vis_point const &rhs)
        {
            return lhs.dist_ < rhs.dist_;
        }

        friend bool operator>(vis_point const &lhs,
                              vis_point const &rhs)
        {
            return !(lhs < rhs);
        }

        friend std::ostream& operator<<(std::ostream &out,
                                        vis_point const &input)
        {
            out<<input.img_name_<<" : "
              <<input.dist_<<" : "
             <<input.kp_.pt;

            return out;
        }

        U dist_;
        std::string img_name_;
        cv::KeyPoint kp_;
    };

    using vis_points = std::vector<vis_point>;

    std::vector<vis_points>
    find_top_results(features_indexer const &fi,
                     arma::Mat<U> const &vocab) const
    {
        std::vector<std::string> img_names;
        fi.read_image_name(img_names);
        //std::cout<<"img size : "<<img_names.size()<<"\n";
        std::cout<<"vocab size : "<<vocab.n_rows<<", "
                <<vocab.n_cols<<"\n";
        std::vector<vis_points> results(vocab.n_cols);
        for(size_t i = 0; i != img_names.size(); ++i){
            cv::Mat img_features;
            std::vector<cv::KeyPoint> keypoints;
            fi.read_image_features(img_features, static_cast<int>(i));
            fi.read_keypoints(keypoints, static_cast<int>(i));
            //std::cout<<img_names[i]<<" : "<<keypoints.size()<<"\n";
            //std::cout<<"img features size : "<<img_features.rows
            //        <<":"<<img_features.cols<<"\n";
            for(int j = 0; j != img_features.rows; ++j){
                arma::Col<U> arma_features(img_features.cols);
                auto *ptr = img_features.ptr<T>(j);
                for(int k = 0; k != img_features.cols; ++k){
                    arma_features(k) = static_cast<U>(ptr[k]);
                }
                auto const distances = euclidean_dist(arma_features,
                                                      vocab);
                //std::cout<<"distances : "<<distances<<std::endl;
                for(arma::uword m = 0; m != vocab.n_cols; ++m){
                    update_vis(distances[m], img_names[i],
                               keypoints[j], results[m]);
                }
            }
        }

        return results;
    }

private:    
    arma::Mat<U> euclidean_dist(arma::Col<U> const &x,
                                arma::Mat<U> const &y) const
    {
        return arma::sqrt(arma::sum
                          (arma::square(y.each_col() - x)));
    }

    void tile(std::vector<cv::Mat> const &src, cv::Mat &dst,
              int grid_x, int grid_y) const
    {
        // patch size
        int const width  = dst.cols/grid_x;
        int const height = dst.rows/grid_y;
        // iterate through grid
        int k = 0;
        for(int i = 0; i < grid_y; i++) {
            for(int j = 0; j < grid_x; j++) {
                Mat s = src[k++];
                cv::resize(s,s,Size(width,height));
                s.copyTo(dst(cv::Rect(j*width,i*height,width,height)));
            }
        }
    }

    void update_vis(U distance, std::string const &img_name,
                    cv::KeyPoint kp, vis_points &vis) const
    {
        if(vis.size() < 16){
            vis.emplace_back(distance, img_name, kp);
            if(vis.size() == 16){
                std::sort(std::begin(vis), std::end(vis));
            }
        }else{
            auto it = std::find_if(std::begin(vis), std::end(vis),
                                   [=](vis_point const &val)
            {
                return val.dist_ > distance;
            });
            if(it != std::end(vis)){
                vis.emplace(it, distance, img_name, kp);
                vis.pop_back();
            }
        }
    }

};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // OCV_CBIR_VISUALIZE_FEATURE_HPP
