#include "command_prompt_utility.hpp"

namespace ocv{

namespace cmd{

boost::program_options::variables_map
parse_command_line(int argc, char **argv)
{
    using namespace boost::program_options;

    try
    {
        options_description desc{"Options"};
        desc.add_options()
                ("help,h", "Help screen")
                ("image,i", value<std::string>(), "Image to process")
                ("video,v", value<std::string>(), "Video to process");

        notify(vm);
        //parse the command line and store it in containers
        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")){
            std::cout << desc << std::endl;
        }

        return vm;
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    return {};
}

}

}
