#ifndef MEASURE_HPP
#define MEASURE_HPP

#include <chrono>

/*! \file measure.hpp
    \brief implement the class/function to measure the\n
    times of programs
*/

/*!
 *  \addtogroup ocv
 *  @{
 */
namespace ocv{

/*!
 *  \addtogroup time
 *  @{
 */
namespace time{

template<typename ToDuaration = std::chrono::milliseconds,
         typename Clock = std::chrono::system_clock>
struct measure
{
    template<typename F, typename ...Args>
    static typename ToDuaration::rep execution(F func, Args&&... args)
    {
        auto start = Clock::now();
        func(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<ToDuaration>
                            (Clock::now() - start);
        return duration.count();
    }

    template<typename F, typename ...Args>
    static ToDuaration duration(F func, Args&&... args)
    {
        auto start = Clock::now();
        func(std::forward<Args>(args)...);
        return std::chrono::duration_cast<ToDuaration>
                (std::chrono::system_clock::now() - start);
    }
};

} /*! @} End of Doxygen Groups*/

} /*! @} End of Doxygen Groups*/

#endif // MEASURE_HPP

