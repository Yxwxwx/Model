#include "ci/kh.hpp"
#include "ci/sc.hpp"
#include "integral/integral.hpp"
#include <iostream>

int main()
{

    integral::Integral<> MO_INTEGRAL("../example/c2_cas88/FCIDUMP");
    ci::SlaterCondon<double> SC("../example/c2_cas88/electron_configurations.txt");
    SC.kernel(MO_INTEGRAL, 6, 6);
    return 0;
}