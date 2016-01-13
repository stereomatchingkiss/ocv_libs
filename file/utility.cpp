#include "utility.hpp"

#include <functional>

namespace ocv{

namespace file{

namespace{

std::vector<std::string>
get_directory_info(std::string const &dir,
                   std::function<bool(boost::filesystem::path const&)> func)
{
    using namespace boost::filesystem;

    path info(dir);
    std::vector<std::string> result;
    if(is_directory(info)){
        directory_iterator it{info};
        while(it != directory_iterator{}){
            if(func(it->path())){
                result.emplace_back(it->path().filename().generic_string());
            }
            ++it;
        }
    }

    return result;
}

}

std::vector<std::string>
get_directory_files(std::string const &dir)
{
    using namespace boost::filesystem;
    return get_directory_info(dir, [](path const &p)
    {
        return is_regular_file(p);
    });
}

std::vector<std::string> get_directory_folders(std::string const &dir)
{
    using namespace boost::filesystem;
    return get_directory_info(dir, [](path const &p)
    {
        return is_directory(p);
    });
}

size_t ocv::file::get_minimum_file_size(const std::string &dir)
{
    auto const folders = get_directory_folders(dir);
    size_t min_data = std::numeric_limits<size_t>::max();
    for(auto const &folder : folders){
        auto const destination = dir + "/" + folder;
        size_t const size =
                get_directory_file_size(destination);
        if(min_data > size){
            min_data = size;
        }
    }
    if(min_data == std::numeric_limits<size_t>::max()){
        min_data = 0;
    }

    return min_data;
}

size_t get_directory_file_size(const std::string &dir)
{
    using namespace boost::filesystem;

    path const info(dir);
    size_t size = 0;
    if(is_directory(info)){
        directory_iterator it{info};
        for(; it != directory_iterator{}; ++it){
            if(is_regular_file(it->path())){
                ++size;
            }
        }
    }

    return size;
}

}

}
