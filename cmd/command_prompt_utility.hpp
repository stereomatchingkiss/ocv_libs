#ifndef OCV_COMMAND_PROMPT_UTILITY_HPP
#define OCV_COMMAND_PROMPT_UTILITY_HPP

#include <boost/program_options.hpp>

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

boost::program_options::variables_map
parse_command_line(int argc, char **argv);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // COMMAND_PROMPT_UTILITY_HPP

