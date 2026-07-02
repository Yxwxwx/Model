#include "ci/sc.hpp"
#include "integral/integral.hpp"
#include <format>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << std::format(
            "Usage: {} <fcidump> [ncas] [nelec] [twos=0] [nroots=2]\n",
            argv[0]);
        return 1;
    }

    std::string fcidump = argv[1];
    size_t ncas   = argc > 2 ? std::stoul(argv[2]) : 8;
    size_t nelec  = argc > 3 ? std::stoul(argv[3]) : 8;
    size_t twos   = argc > 4 ? std::stoul(argv[4]) : 0;
    size_t nroots = argc > 5 ? std::stoul(argv[5]) : 2;

    std::cout << std::format("FCIDUMP: {}\n", fcidump);
    std::cout << std::format("CAS({}, {}), 2Sz={}, nroots={}\n",
                             ncas, nelec, twos, nroots);

    integral::Integral<> mo_integral(fcidump);
    ci::SlaterCondon<double> sc(ncas, nelec, twos);
    sc.kernel(mo_integral, nroots);

    return 0;
}
