#include "features_indexer.hpp"

#include <iostream>

namespace ocv{

namespace cbir{

features_indexer::features_indexer(std::string const &file_name)
    : buffer_size_{20},
      cur_buffer_size_{0},
      feature_row_offset_{0},
      features_size_{0},
      h5io_(cv::hdf::open(file_name)),
      image_row_offset_{0},
      name_size_{0}
{    
    if(h5io_->hlexists("image_name")){
        auto const size = h5io_->dsgetsize("image_name");
        image_row_offset_ = size[0];
        name_size_ = size[1];
    }

    if(h5io_->hlexists("features")){
        auto const size = h5io_->dsgetsize("features");
        feature_row_offset_ = size[0];
        features_size_ = size[1];
    }
}

features_indexer::~features_indexer()
{
    flush();
}

void features_indexer::add_features(const std::string &image_name,
                                    cv::Mat features)
{
    if(features.empty()){
        return;
    }

    if(image_name.size() == name_size_){
        img_names_ += image_name;
    }else{
        auto temp = image_name;
        temp.resize(name_size_);
        img_names_ += temp;
    }

    if(index_.empty()){
        index_.emplace_back(0);
        index_.emplace_back(features.rows);
    }else{
        auto const last_size = index_.back();
        index_.emplace_back(last_size);
        index_.emplace_back(last_size + features.rows);
    }

    //vconcat is an expensive operations
    if(!features_.empty()){
        cv::vconcat(features_.clone(), features, features_);
    }else{
        cv::vconcat(features, features_);
    }

    ++cur_buffer_size_;
    if(cur_buffer_size_ >= buffer_size_){
        flush();
    }
}

void features_indexer::
create_dataset(int name_size,
               int features_size,
               int features_type)
{
    if(!h5io_->hlexists("image_name")){
        int const chunks[] = {10, name_size};
        name_size_ = name_size;
        h5io_->dscreate(cv::hdf::HDF5::H5_UNLIMITED,
                        name_size_,
                        CV_8S, "image_name",
                        cv::hdf::HDF5::H5_NONE,
                        chunks);
    }
    if(!h5io_->hlexists("index")){
        int const chunks[] = {100, 2};
        h5io_->dscreate(cv::hdf::HDF5::H5_UNLIMITED,
                        2,
                        CV_32S, "index",
                        cv::hdf::HDF5::H5_NONE,
                        chunks);
    }
    if(!h5io_->hlexists("features")){
        int const chunks[] = {100, features_size};
        features_size_ = features_size;
        h5io_->dscreate(cv::hdf::HDF5::H5_UNLIMITED,
                        features_size,
                        features_type,"features",
                        cv::hdf::HDF5::H5_NONE,
                        chunks);
    }
}

void features_indexer::flush()
{
    if(cur_buffer_size_ != 0){
        int const f_offset[] = {feature_row_offset_, 0};
        h5io_->dsinsert(features_, "features", f_offset);

        int const im_offset[] = {image_row_offset_, 0};
        cv::Mat_<char> im_name(static_cast<int>(img_names_.size()/name_size_),
                               name_size_,
                               const_cast<char*>(&img_names_[0]),
                name_size_ * sizeof(char));
        h5io_->dsinsert(im_name, "image_name", im_offset);

        int const i_offset[] = {image_row_offset_, 0};
        cv::Mat_<int> index(static_cast<int>(index_.size() / 2), 2,
                            &index_[0], 2 * sizeof(int));
        h5io_->dsinsert(index, "index", i_offset);

        feature_row_offset_ += features_.rows;
        image_row_offset_ += static_cast<int>(index_.size()) / 2;

        cur_buffer_size_ = 0;
        features_ = cv::Mat();
        img_names_.clear();
        index_.clear();
    }else{
        return;
    }
}

std::vector<int> features_indexer::get_features_dimension() const
{
    return get_dimension("features");
}

std::vector<int> features_indexer::get_index_dimension() const
{
    return get_dimension("index");
}

std::vector<int> features_indexer::get_names_dimension() const
{
    return get_dimension("image_name");
}

void features_indexer::read_data(cv::InputOutputArray &features,
                                 cv::InputOutputArray &features_index,
                                 std::vector<std::string> &image_names,
                                 int img_begin, int img_end) const
{
    int const i_offset[] = {img_begin, 0};
    int const i_count[] = {img_end - img_begin + 1, 2};
    h5io_->dsread(features_index, "index", i_offset, i_count);

    int const im_offset[] = {img_begin, 0};
    int const im_count[] = {img_end - img_begin + 1, name_size_};
    cv::Mat_<char> names;
    h5io_->dsread(names, "image_name", im_offset, im_count);
    image_names.clear();
    for(int row = 0; row != names.rows; ++row){
        auto *ptr = names.ptr<char>(row);
        std::string name(ptr, ptr + name_size_);
        image_names.emplace_back(std::move(name));
    }

    auto const f_index = features_index.getMat_();
    int const f_offset[] = {f_index.at<int>(0,0), 0};
    int const f_count[] = {f_index.at<int>(f_index.rows-1,1) -
                           f_index.at<int>(0,0),
                           features_size_};
    h5io_->dsread(features, "features", f_offset, f_count);        
}

void features_indexer::set_buffer_size(size_t value)
{
    buffer_size_ = value;
}

std::vector<int> features_indexer::
get_dimension(const std::string &label) const
{
    if(h5io_->hlexists(label)){        
        return h5io_->dsgetsize(label);
    }

    return {};
}


}

}

