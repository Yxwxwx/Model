#include <format>
#include <iostream>

#include "integral.hpp"
#include "plane.hpp"

int main() {
  // GTO
  std::vector<double> exps = {0.48};
  std::vector<double> coeffs = {1.0};
  std::array<int, 3> lmn = {1, 0, 0};
  std::array<double, 3> center = {0.0, 0.0, 0.0};
  std::vector<std::pair<int, std::array<double, 3>>> q{{1, {0.0, 0.0, 0.0}}};

  // normalization(coeffs, lmn, exps);
  // PW
  const std::array<double, 3> k{1.0, 0.0, 0.0};
  const auto ovlp = contracted_overlap_gp(exps, coeffs, lmn, center, k);
  std::cout << std::format("<G|ovlp|PW> = {:.15f} {:+.15f}i", ovlp.real(),
                           ovlp.imag())
            << '\n';

  const auto kin = contracted_kinetic_gp(exps, coeffs, lmn, center, k);
  std::cout << std::format("<G|kin|PW> = {:.15f} {:+.15f}i", kin.real(),
                           kin.imag())
            << '\n';

  std::array test_cases{std::pair{0, 0.5 + 0.0i},  std::pair{2, 0.5 + 0.0i},
                        std::pair{0, 25.0 + 0.0i}, std::pair{3, 25.0 + 0.0i},
                        std::pair{0, 0.0 + 1.0i},  std::pair{1, 0.0 + 5.0i},
                        std::pair{0, 1.2 + 0.5i},  std::pair{2, 15.0 + 10.0i}};
  for (auto& i : test_cases) {
    const auto val = boys(i.first, i.second);
    // std::cout << std::format("m={}, T=({},{}) => {:.15f}{:+.15f}i\n",
    // i.first,
    //                          i.second.real(), i.second.imag(), val.real(),
    //                          val.imag());
  }

  const auto nuc = contracted_nuclear_gp(exps, coeffs, lmn, center, k, q);
  std::cout << std::format("<G|nuc|PW> = {:.15f} {:+.15f}i", nuc.real(),
                           nuc.imag())
            << '\n';

  const auto eri =
      contracted_eri_gpgg(exps, coeffs, lmn, center, k, exps, coeffs, lmn,
                          center, exps, coeffs, lmn, center);
  std::cout << std::format("(GP|GG) = {:.15f} {:+.15f}i", eri.real(),
                           eri.imag())
            << '\n';
  std::cout << std::endl;
}
