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

}

}
