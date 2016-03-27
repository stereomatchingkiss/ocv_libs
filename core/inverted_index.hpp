#ifndef OCV_CORE_INVERTED_INDEX_HPP
#define OCV_CORE_INVERTED_INDEX_HPP

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
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

    void erase_key(Key const &key)
    {
        index_.erase(key);
    }

    void erase_value(Key const &key,
                     Value const &value)
    {
        auto it = index_.find(key);
        if(it != std::end(index_)){
            auto &vec = it->second;
            auto it2 = std::find(std::begin(vec),
                                 std::end(vec), value);
            if(it2 != std::end(vec)){
                vec.erase(it2);
            }
        }
    }

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
        (it->second).emplace_back(val);
    }

    void load(std::string const &name)
    {
        std::ifstream in_f(name);
        boost::archive::text_iarchive ia(in_f);
        serialize(ia, *this, 0);
    }

    void save(std::string const &name) const
    {
        std::ofstream of(name);
        boost::archive::text_oarchive oa(of);
        serialize(oa, *this, 0);
    }

private:
    friend class boost::serialization::access;

    template<typename Archive>
    friend void serialize(Archive &ar, inverted_index &val,
                          const unsigned int)
    {
        ar & val;
    }

    std::map<Key, doc_type> index_;
};

} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_INVERTED_INDEX_HPP
