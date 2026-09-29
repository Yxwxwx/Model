#include <format>
#include <iostream>
#include <tuple>

#include "integral.hpp"
#include "plane.hpp"

int main() {
  using namespace std::complex_literals;
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
  std::cout << std::format(
                   "<G|ovlp|PW> = {:.15f} {:+.15f}i", ovlp.real(), ovlp.imag())
            << '\n';

  const auto kin = contracted_kinetic_gp(exps, coeffs, lmn, center, k);
  std::cout << std::format(
                   "<G|kin|PW> = {:.15f} {:+.15f}i", kin.real(), kin.imag())
            << '\n';

  // References: 80-digit quadrature of integral_0^1 x^(2m) exp(-T*x*x) dx.
  const std::tuple<int, std::complex<double>, std::complex<double>>
      test_cases[] = {
          {0, 0.0, 1.0},
          {6, 0.0, 0.07692307692307693},
          {12, 0.5, 0.025191805984945876},
          {0, -9.99, 1157.8484463885602},
          {3, -9.9, 788.8449163296597},
          {6, -9.82611111111111, 588.8643343151275},
          {0, -8.0 + 5.0i, -50.055693699618544 + 157.52782955876867i},
          {6, -8.0 + 5.0i, -4.885525587721108 + 101.17625783533204i},
          {0, 9.9i, 0.1784126519503087 - 0.2448827899068432i},
          {6, 9.9i, -0.03891764542412523 - 0.023684630120627083i},
          {0, 9.999999, 0.2802474045167433},
          {6, 9.999999, 4.1184840443738106e-05},
          {6, 10.000001, 4.118479144347039e-05},
          {0, -10.000001, 1168.2315064916772},
          {6, -10.000001, 693.2342141436573},
          {3, 25.0, 2.1269445903632173e-05},
          {2, 15.0 + 10.0i, 4.8468769353180505e-05 - 0.0004792366784814511i},
          // Both sides of the old negative and current positive |T|=50
          // boundary, and both sides of the |Im(T)|=10 boundary. Inputs are
          // exact binary values.
          {0, -50.125, 5.9206911480216891415e19},
          {0, -50.0, 5.2381917621841878397e19},
          {0, -49.875, 4.6343961304687783851e19},
          {12, 49.875, 4.0893792793565228612e-14},
          {12, 50.0, 3.9634072238087975146e-14},
          {12, 50.125, 3.8416158136607834621e-14},
          {4,
           -0.125 + 9.875i,
           -0.04048166827176992662 - 0.03803472690146052902i},
          {0, -0.125 + 10.0i, 0.16799182411816859685 - 0.24776268102942249764i},
          {12, 10.0i, -0.03235076131810942409 - 0.00284516193633856555i},
          {0, -5.0 + 10.125i, -5.8569426024977350958 - 3.1568180799173773102i},
      };
  for (const auto& [m, T, reference] : test_cases) {
    const auto val = boys(m, T);
    if (!(std::abs(val - reference) <= 5e-15 * std::abs(reference))) {
      std::cerr << "Boys regression failed: m=" << m << ", T=" << T
                << ", value=" << val << ", reference=" << reference << '\n';
      return 1;
    }
  }

  // Guard digits must survive the negative-T series used by GPPG.
  const std::tuple<int, std::complex<long double>, std::complex<long double>>
      extended_cases[] = {
          {0, -50.0L, 5.23819176218418783969528623993676911e19L},
          {4, -100.0L, 1.29817502005873641382376810515838997e41L},
          {0, -500.0L, 1.40500004204994543488568350533377904e214L},
          {4,
           {40.0L, 5.0L},
           {2.93980099921597456059827968776446092e-7L,
            -1.84145881219013436959559365645543722e-7L}},
      };
  for (const auto& [m, T, reference] : extended_cases) {
    const auto val = boys(m, T);
    const auto tolerance =
        std::max(1e-17L, 32 * std::numeric_limits<long double>::epsilon());
    if (!(std::abs((val - reference) / reference) <= tolerance)) {
      std::cerr << "Extended-precision Boys regression failed: m=" << m
                << ", T=" << T << ", value=" << val << '\n';
      return 1;
    }
  }
  std::cout << "PASS: Boys series, branch boundaries, and extended precision\n";

  const auto nuc = contracted_nuclear_gp(exps, coeffs, lmn, center, k, q);
  std::cout << std::format(
                   "<G|nuc|PW> = {:.15f} {:+.15f}i", nuc.real(), nuc.imag())
            << '\n';

  const auto eri = contracted_eri_gpgg(
      exps,
      coeffs,
      lmn,
      center,
      k,
      exps,
      coeffs,
      lmn,
      center,
      exps,
      coeffs,
      lmn,
      center);
  std::cout << std::format(
                   "(GP|GG) = {:.15f} {:+.15f}i", eri.real(), eri.imag())
            << '\n';
  std::cout << '\n';
}
