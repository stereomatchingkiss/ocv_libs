#include "gradient_checking.hpp"

namespace ocv{

namespace ml{

gradient_checking::gradient_checking() :
    epsillon_(0.001)
{

}

void gradient_checking::set_epsillon(double epsillon)
{
    epsillon_ = epsillon;
}

}}
