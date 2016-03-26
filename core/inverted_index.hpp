#ifndef OCV_CORE_INVERTED_INDEX_HPP
#define OCV_CORE_INVERTED_INDEX_HPP

#include <map>
#include <vector>

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/**
 *inverted index based on stl
 *@tparam Key key of the inverted index
 *@tparam Value value of the vector of inverted index
 */
template<typename Key, typename Value>
class inverted_index
{
public:
    using doc_type = std::vector<Value>;
    using map_type = std::map<Key, doc_type>;

    void erase_key(Key const &key);
    void erase_value(Key const &key,
                     Value const &value);

    map_type::const_iterator
    find(Key const &key) const
    {
        return index_.find(key);
    }

    map_type::iterator
    find(Key const &key)
    {
        return index_.find(key);
    }

    void insert(Key const &key, Value const &val)
    {
        auto it = index_.find(key);
        if(it != std::end(index_)){
            it = index_.insert({val}).first;
        }
        it->second.emplace_back(val);
    }

private:
    std::map<Key, doc_type> index_;
};

} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_INVERTED_INDEX_HPP
