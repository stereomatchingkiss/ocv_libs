#ifndef OCV_CORE_INVERTED_INDEX_HPP
#define OCV_CORE_INVERTED_INDEX_HPP

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <iterator>
#include <map>
#include <type_traits>
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
    using value_type = Value;

    typename map_type::const_iterator
    begin() const
    {
        return index_.begin();
    }

    typename map_type::iterator
    begin()
    {
        return index_.begin();
    }

    bool empty() const
    {
        return index_.empty();
    }

    typename map_type::const_iterator
    end() const
    {
        return index_.end();
    }

    typename map_type::iterator
    end()
    {
        return index_.end();
    }

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

    typename map_type::const_iterator
    find(Key const &key) const
    {
        return index_.find(key);
    }

    typename map_type::iterator
    find(Key const &key)
    {
        return index_.find(key);
    }

    void insert(Key const &key, Value const &val)
    {
        auto it = index_.find(key);
        if(it == std::end(index_)){
            it = index_.insert({key, {}}).first;
        }
        (it->second).emplace_back(val);
    }

    template<typename InputIter>
    void insert(Key const &key, InputIter beg,
                InputIter end)
    {
        auto it = index_.find(key);
        if(it == std::end(index_)){
            it = index_.insert({key, {}}).first;
        }
        (it->second).insert(std::end(it->second),
                            beg, end);
    }

    void load(std::string const &name)
    {
        std::ifstream in_f(name);
        boost::archive::text_iarchive ia(in_f);
        serialize(ia, *this, 0);
    }

    void save(std::string const &name)
    {
        std::ofstream of(name);
        boost::archive::text_oarchive oa(of);
        serialize(oa, *this, 0);
    }

    size_t size() const
    {
        return index_.size();
    }

private:
    friend class boost::serialization::access;

    template<typename Archive>
    friend void serialize(Archive &ar, inverted_index &val,
                          const unsigned int)
    {
        ar & val.index_;
    }

    std::map<Key, doc_type> index_;
};

} /*! @} End of Doxygen Groups*/

#endif // OCV_CORE_INVERTED_INDEX_HPP
