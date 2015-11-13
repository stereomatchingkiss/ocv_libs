#ifndef OCV_COMMAND_PROMPT_UTILITY_HPP
#define OCV_COMMAND_PROMPT_UTILITY_HPP

#include <boost/program_options.hpp>

#include <functional>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup cmd
 *  @{
 */
namespace cmd{

/**
 * Count the commands number input by the users
 * @param input key value pair of the commands
 * @param commands keys of the input
 * @return commands number being activated
 */
size_t active_commands_size(boost::program_options::variables_map const &input,
                            std::vector<std::string> const &commands);

/**
 * Default command line parsers, it can parse following commands
 * --help,h
 * --video,v
 * --video_folder,V
 * --image,i
 * --image_folder,I
 * @param argc size of the command line arguments
 * @param argv contents of the command line arguments
 * @return key value pair after parsed
 */
std::pair<boost::program_options::variables_map, std::vector<std::string>>
default_command_line_parser(int argc, char **argv);

/**
 * Default command line parsers, it can parse following commands
 * --help,h
 * --video,v
 * --video_folder,V
 * --image,i
 * --image_folder,I
 * @param argc size of the command line arguments
 * @param argv contents of the command line arguments
 * @param function Can accept the options_description thus
 * enable the users to finetune the options
 * @return key value pair after parsed
 */
std::pair<boost::program_options::variables_map, std::vector<std::string>>
default_command_line_parser(int argc, char **argv,
                            std::function<void(boost::program_options::options_description&)> function);

/**
 * Check the commands is mutual exclusive or not, in other words,
 * the activate commands must be <= 1
 * @param input key value pairs of the commands
 * @param commands key of the input
 * @return true if mutual exclusive and vice versa
 */
bool is_mutual_exclusive(boost::program_options::variables_map const &input,
                         std::vector<std::string> const &commands);

/**
 * Overload of is_mutual_exclusive, but this one accept a pair value
 * @return  true if mutual exclusive and vice versa
 */
bool is_mutual_exclusive(std::pair<boost::program_options::variables_map,
                         std::vector<std::string>> const &input);

/**
 * All of the commands must be mutually exclusive and
 * at least one of the commands must be specified
 * @param input key value pairs of the commands
 * @param commands key of the input
 * @return true if mutual exclusive and vice versa
 */
bool require_mutual_exclusive(boost::program_options::variables_map const &input,
                              std::vector<std::string> const &commands);

/**
 * Overload of require_mutual_exclusive, but this one accept a pair value
 * @return  true if require mutual exclusive and vice versa
 */
bool require_mutual_exclusive(std::pair<boost::program_options::variables_map,
                              std::vector<std::string>> const &commands);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // COMMAND_PROMPT_UTILITY_HPP

