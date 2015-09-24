#include "crop_images.hpp"

#include "for_each.hpp"

#include <boost/filesystem.hpp>

#include <opencv2/highgui.hpp>

#include <iomanip>
#include <sstream>

namespace ocv{

/**
 * @brief crop the image into non-overlap small block
 * @param directory directory of the image
 * @param img_name name of the image
 * @param size block size of the image
 * @return the images after cropped
 */
std::vector<cv::Mat> crop_image(std::string const &directory,
                                std::string const &img_name,
                                cv::Size2i const &size)
{    
    cv::Mat const Img = cv::imread(directory + "/" + img_name);
    std::vector<cv::Mat> result;
    if(!Img.empty()){
        auto func = [&](int, int, cv::Mat const &input)
        {
            result.emplace_back(input);
        };
        ocv::for_each_block(Img, size, func, size);
    }

    return result;
}

/**
 * @brief crop all of the images of the directory into non-overlap small block
 * @param directory directory of the image
 * @param size block size of the image
 * @return the images after cropped
 */
std::vector<cv::Mat> crop_directory_images(std::string const &directory,
                                           cv::Size2i const &size)
{
    using namespace boost::filesystem;

    path info(directory);
    std::vector<cv::Mat> result;
    if(is_directory(info)){
        directory_iterator it{info};
        while(it != directory_iterator{}){
            if(is_regular_file(*it)){
                auto const File = it->path().filename().generic_string();
                auto files = crop_image(directory, File, size);
                result.insert(std::end(result), std::begin(files), std::end(files));
            }
            ++it;
        }
    }

    return result;
}

}
