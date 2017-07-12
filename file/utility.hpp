#ifndef OCV_FILE_UTILITY_HPP
#define OCV_FILE_UTILITY_HPP

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
 * Get the files of the directory with/without recursively get
 * the other files of the folder underlying the folder
 * @param dir directory want to get files
 * @param recursive true, get the files recursive and vice versa
 * @return file names of dir(without path but with suffix)
 */
std::vector<std::string>
get_directory_files(std::string const &dir, bool recursive = false);

/**
 * Get the boost::filesystem::path of the directory with/without
 * recursively get the other files of the folder underlying the folder
 * @param dir directory want to get files
 * @param recursive true, get the paths recursive and vice versa
 * @return paths under the directory
 */
std::vector<boost::filesystem::path>
get_directory_path(std::string const &dir, bool recursive = false);

/**
 * Overload of get_directory_path
 * @param dir directory want to get files
 * @param valid_extension Type of the files want to get
 * @param case_sensitive self explain
 * @param recursive true, get the paths recursive and vice versa
 * @return paths under the directory
 * @code
 * auto const paths = get_directory_path("/home/myHome/fishering_monitor", {".jpg", ".jpeg", ".png", ".bmp"});
 * @endcode
 */
std::vector<boost::filesystem::path>
get_directory_path(std::string const &dir, std::vector<std::string> const &valid_extension,
                   bool case_sensitive = false,
                   bool recursive = false);

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

