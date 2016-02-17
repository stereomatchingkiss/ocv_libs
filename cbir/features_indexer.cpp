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
        image_row_offset_ = h5io_->dsgetsize("image_name")[0];
    }

    if(h5io_->hlexists("features")){
        feature_row_offset_ = h5io_->dsgetsize("features")[0];
    }
}

features_indexer::~features_indexer()
{
    flush();
}

void features_indexer::add_features(const std::string &image_name,
                                    cv::Mat features)
{
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
        cv::vconcat(features, features_.clone(), features_);
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

void features_indexer::
read_features(cv::Mat &inout, const std::string &image_name) const
{

}

void features_indexer::read_features(cv::InputOutputArray &features,
                                     cv::InputOutputArray &index,
                                     int begin_index, int end_index) const
{
    int const i_offset[] = {begin_index, 0};
    int const i_count[] = {end_index - begin_index, 2};
    h5io_->dsread(index, "index", i_offset, i_count);


}

void features_indexer::
read_image_name(std::vector<std::string> &inout) const
{

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

