#include "utility.hpp"

#include <algorithm>
#include <iostream>
#include <functional>
#include <set>

namespace ocv{

namespace file{

namespace{

std::vector<boost::filesystem::path>
get_directory_path_impl(const std::string &dir,
                        std::function<bool(boost::filesystem::path const&)> func,
                        bool recursive = false)
{
    using namespace boost::filesystem;

    path info(dir);
    std::vector<path> result;
    if(is_directory(info)){
        if(recursive){
            recursive_directory_iterator beg(dir), end;
            for(; beg != end; ++beg){
                if(func(beg->path())){
                    result.emplace_back(beg->path());
                }
            }
        }else{
            directory_iterator beg{dir}, end;
            for(; beg != end; ++beg){
                if(func(beg->path())){
                    result.emplace_back(beg->path());
                }
            }
        }
    }

    return result;
}

template<typename T>
std::vector<boost::filesystem::path> filter_by_extension(T const &filters,
                                                         std::vector<boost::filesystem::path> &paths)
{
    using namespace boost::filesystem;
    std::vector<path> results;
    for(path &p : paths){
        if(filters.find(p.extension().string()) != std::end(filters)){
            results.emplace_back(std::move(p));
        }
    }

    return results;
}

} //nameless namespace

std::vector<std::string>
get_directory_files(std::string const &dir, bool recursive)
{
    using namespace boost::filesystem;

    auto func = [](path const &p)
    {
        return is_regular_file(p);
    };

    auto const paths = get_directory_path_impl(dir, func, recursive);
    std::vector<std::string> result;
    std::transform(std::begin(paths), std::end(paths),
                   std::back_inserter(result), [](path const &p)
    {
        return p.filename().generic_string();
    });

    return result;
}

std::vector<std::string> get_directory_folders(std::string const &dir)
{
    using namespace boost::filesystem;

    auto const paths = get_directory_path_impl(dir, [](path const &p)
    {
            return is_directory(p);
    });

    std::vector<std::string> result;
    std::transform(std::begin(paths), std::end(paths),
                   std::back_inserter(result), [](path const &p)
    {
        return p.filename().generic_string();
    });

    return result;
}

size_t get_minimum_file_size(const std::string &dir)
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

std::vector<boost::filesystem::path>
get_directory_path(const std::string &dir, bool recursive)
{
    using namespace boost::filesystem;

    return get_directory_path_impl(dir, [](path const&){ return true;}, recursive);
}

std::vector<boost::filesystem::path> get_directory_path(const std::string &dir,
                                                        const std::vector<std::string> &valid_extension,
                                                        bool case_sensitive, bool recursive)
{
    using namespace boost::filesystem;

    std::vector<path> paths = get_directory_path(dir, recursive);
    if(case_sensitive){
        std::set<std::string> const filters(std::begin(valid_extension), std::end(valid_extension));
        return filter_by_extension(filters, paths);
    }else{
        struct icase
        {
            //less effective but easier to implement
            bool operator()(std::string a, std::string b) const
            {
                std::transform(std::begin(a), std::begin(a), std::begin(a), [](char c){ return std::tolower(c); });
                std::transform(std::begin(b), std::begin(b), std::begin(b), [](char c){ return std::tolower(c); });
                return a < b;
            }
        };

        std::set<std::string, icase> const filters(std::begin(valid_extension), std::end(valid_extension));
        return filter_by_extension(filters, paths);
    }
}

}

}
