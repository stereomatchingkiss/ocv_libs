#include "command_prompt_utility.hpp"

#include <iostream>
#include <numeric>

namespace ocv{

namespace cmd{

using variables_map = boost::program_options::variables_map;

size_t active_commands_size(variables_map const &input,
                            std::vector<std::string> const &commands)
{
    return std::accumulate(std::begin(commands), std::end(commands), size_t(0),
                           [&](size_t init, std::string const &cmd)
    {
        return (input.count(cmd) != 0 && !input[cmd].empty()) ?
                    init + 1 : init;
    });
}

std::pair<variables_map, std::vector<std::string>>
default_command_line_parser(int argc, char **argv)
{
    using namespace boost::program_options;

    return default_command_line_parser(argc, argv,
                                       [](options_description&){});
}

std::pair<variables_map, std::vector<std::string>>
default_command_line_parser(int argc, char **argv,
                            std::function<void(boost::program_options::options_description&)> function)
{
    using namespace boost::program_options;

    try
    {
        options_description desc{"Options"};
        desc.add_options()
                ("help,h", "Help screen")
                ("image,i", value<std::string>(), "Image to process")
                ("image_folder,I", value<std::string>(), "Specify the folder of the images to process")
                ("output_folder,o", value<std::string>(), "Specify the output folder")
                ("random_size,r", value<size_t>(), "Specify the random size")
                ("video,v", value<std::string>(), "Video to process")
                ("video_folder,V", value<std::string>(), "Specify the folder of the videos to process");

        function(desc);
        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);

        notify(vm);

        if (vm.count("help")){
            std::cout << desc << std::endl;
        }

        std::vector<std::string> commands;
        for(auto const &data : desc.options()){
           commands.emplace_back(data->long_name());
        }

        return {vm, commands};
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return {};
}

bool is_mutual_exclusive(variables_map const &input,
                         std::vector<std::string> const &commands)
{
    if(commands.size() != 0){
        if(active_commands_size(input, commands) <= 1){
            return true;
        }

        std::cout<<"The commands [";
        std::copy(std::begin(commands), std::end(commands) - 1,
                  std::ostream_iterator<std::string>(std::cout, ","));
        std::cout<<*(commands.end()-1)<< "] are mutually eclusive"<< std::endl;

        return false;
    }

    return true;
}

bool is_mutual_exclusive(std::pair<boost::program_options::variables_map,
                         std::vector<std::string>> const &input)
{
    return is_mutual_exclusive(input.first, input.second);
}

bool require_mutual_exclusive(variables_map const &input,
                              std::vector<std::string> const &commands)
{
    if(commands.size() != 0){        
        if(active_commands_size(input, commands) == 1){
            return true;
        }

        std::cout<<"One of the following commands [";
        std::copy(std::begin(commands), std::end(commands) - 1,
                  std::ostream_iterator<std::string>(std::cout, ","));
        std::cout<<*(commands.end()-1)
                << "] must be speficy, and they"
                   " are mutually eclusive"<< std::endl;

        return false;
    }

    return true;
}

bool require_mutual_exclusive(std::pair<boost::program_options::variables_map,
                              std::vector<std::string>> const &commands)
{
    return require_mutual_exclusive(commands.first, commands.second);
}

}

}
