#ifndef OCV_UTILITY_HPP
#define OCV_UTILITY_HPP

#include <boost/filesystem.hpp>

#include <string>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup file
 *  @{
 */
namespace file{

/**
 * Get the files of the directory without recursively get
 * the other files of the folder underlying the folder
 * @param dir directory want to get files
 * @return file names of dir(without path but with suffix)
 */
std::vector<std::string>
get_directory_files(std::string const &dir);

/**
 * Get the file size of the directory without recursively get
 * the other files of the folder underlying the folder
 * @param dir directory want to get files
 * @return file size of the directory
 */
size_t get_directory_file_size(std::string const &dir);

/**
 * Get the directories of the directory without recursively get
 * the other directories of the folder underlying the folder
 * @param dir directory want to get directories
 * @return direcotories of dir(without path but with suffix)
 */
std::vector<std::string>
get_directory_folders(std::string const &dir);

/**
 * Get the minimum file size of the folders under dir
 * without recursive
 * @param dir Directory want to calculate minimum file size
 * @return minimum file size of the folders under dir
 */
size_t get_minimum_file_size(std::string const &dir);

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // UTILITY_HPP

