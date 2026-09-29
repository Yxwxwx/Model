#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <bit>
#include <cstdint>
#include <string>

#include "integral.hpp"

namespace py = pybind11;

namespace {
using Buffer = py::array_t<std::complex<double>, py::array::c_style>;
using Ints = py::array_t<int, py::array::c_style | py::array::forcecast>;
using Reals = py::array_t<double, py::array::c_style | py::array::forcecast>;
using OneWaveKernel = void (*)(
    std::complex<double>*,
    const int*,
    const int*,
    int,
    const int*,
    int,
    const double*,
    const double*);
using TwoWaveKernel = void (*)(
    std::complex<double>*,
    const int*,
    const int*,
    int,
    const int*,
    int,
    const double*,
    const double*,
    const double*);

// Integer exponent bits keep validation valid under fast-math.
bool is_finite(double value) {
  return (std::bit_cast<std::uint64_t>(value) & 0x7ff0000000000000ULL) !=
      0x7ff0000000000000ULL;
}

template <typename Kernel>
Kernel select_precision(const std::string& precision, Kernel low, Kernel high) {
  if (precision == "low") {
    return low;
  }
  if (precision == "high") {
    return high;
  }
  throw py::value_error("precision must be 'low' or 'high'");
}

void validate_wavevector(const Reals& k) {
  if (k.ndim() != 1 || k.size() != 3) {
    throw py::value_error("Expected a 3D wavevector");
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!is_finite(k.data()[axis])) {
      throw py::value_error("Wavevectors must be finite");
    }
  }
}

void validate_input(
    const Buffer& buf,
    const Ints& shls,
    const Ints& atm,
    int natm,
    const Ints& bas,
    int nbas,
    const Reals& env,
    int nshell,
    bool spherical) {
  if (shls.ndim() != 1 || shls.size() != nshell) {
    throw py::value_error("Wrong number of GTO shell indices");
  }
  if (natm < 0 || nbas < 0 || atm.ndim() != 2 || bas.ndim() != 2 ||
      atm.shape(0) != natm || atm.shape(1) != ATM_SLOTS ||
      bas.shape(0) != nbas || bas.shape(1) != BAS_SLOTS || env.ndim() != 1) {
    throw py::value_error("Invalid libcint-compatible atm/bas/env dimensions");
  }
  // Nuclear attraction reads coordinates for every atom, including atoms whose
  // basis functions are not in the selected shells.
  for (int atom = 0; atom < natm; ++atom) {
    const py::ssize_t coord = atm.data()[ATM_SLOTS * atom + PTR_COORD];
    if (coord < 0 || coord + 3 > env.size()) {
      throw py::value_error("Atom coordinates outside env");
    }
  }
  py::ssize_t size = 1;
  for (int index = 0; index < nshell; ++index) {
    const int shell = shls.data()[index];
    if (shell < 0 || shell >= nbas) {
      throw py::value_error("GTO shell index out of range");
    }
    const int* info = bas.data() + BAS_SLOTS * static_cast<py::ssize_t>(shell);
    const int atom = info[ATOM_OF], angular = info[ANG_OF];
    const py::ssize_t nprim = info[NPRIM_OF], nctr = info[NCTR_OF];
    if (atom < 0 || atom >= natm || angular < 0 || nprim <= 0 || nctr <= 0) {
      throw py::value_error("Invalid GTO shell metadata");
    }
    if (spherical && angular > L_MAX) {
      throw py::value_error(
          "Spherical angular momentum must be between 0 and 6");
    }
    if (info[PTR_EXP] < 0 || info[PTR_EXP] + nprim > env.size() ||
        info[PTR_COEFF] < 0 || info[PTR_COEFF] + nprim * nctr > env.size()) {
      throw py::value_error("GTO shell data outside env");
    }
    const py::ssize_t components = spherical
        ? 2LL * angular + 1
        : (static_cast<py::ssize_t>(angular) + 1) * (angular + 2LL) / 2;
    if (components > buf.size() / size / nctr) {
      throw py::value_error("Output buffer is too small for the GTO shells");
    }
    size *= components * nctr;
  }
}
} // namespace

PYBIND11_MODULE(libMixedInt, m) {
  m.doc() =
      "GTO-PW integrals with libcint-compatible atm/bas/env arrays. "
      "precision='low' (default): double; precision='high': long double. "
      "Outputs are complex128; precision selects arithmetic, not an error bound. "
      "Spherical kernels include angular normalization and require no PySCF. "
      "p order is x,y,z; l>=2 order is m=-l,...,+l; supported l=0,...,6.";

  m.def(
      "cart2sph_coeff",
      [](int angular) {
        const auto& coefficients = cart2sph_coeff(angular);
        py::array_t<double> result(
            {(angular + 1) * (angular + 2) / 2, 2 * angular + 1});
        for (std::size_t i = 0; i < coefficients.size(); ++i) {
          result.mutable_data()[i] = static_cast<double>(coefficients[i]);
        }
        return result;
      },
      "Raw Cartesian to normalized real spherical matrix (ncart, 2*l+1). "
      "Includes s/p factors; equivalent to PySCF cart2sph(l, normalized=None).",
      py::arg("angular"));

  const auto bind_one_wavevector = [&](const char* name,
                                       int nshell,
                                       bool spherical,
                                       OneWaveKernel low,
                                       OneWaveKernel high) {
    m.def(
        name,
        [=](Buffer buf,
            const Ints& shls,
            const Ints& atm,
            int natm,
            const Ints& bas,
            int nbas,
            const Reals& env,
            const Reals& k_vector,
            const std::string& precision) {
          const auto kernel = select_precision(precision, low, high);
          validate_input(
              buf, shls, atm, natm, bas, nbas, env, nshell, spherical);
          validate_wavevector(k_vector);
          kernel(
              buf.mutable_data(),
              shls.data(),
              atm.data(),
              natm,
              bas.data(),
              nbas,
              env.data(),
              k_vector.data());
        },
        spherical
            ? "Normalized real spherical shell integrals. GTO axes follow shls; "
              "each has nctr*(2*l+1) components, last axis fastest. "
              "buf must be writable C-contiguous complex128."
            : "Raw Cartesian shell integrals. GTO axes follow shls; "
              "each has nctr*(l+1)*(l+2)/2 components, last axis fastest. "
              "buf must be writable C-contiguous complex128.",
        py::arg("buf").noconvert(),
        py::arg("shls"),
        py::arg("atm"),
        py::arg("natm"),
        py::arg("bas"),
        py::arg("nbas"),
        py::arg("env"),
        py::arg("k_vector"),
        py::arg("precision") = "low");
  };
  bind_one_wavevector(
      "int1e_ovlp_cart",
      1,
      false,
      int1e_ovlp_cart<double>,
      int1e_ovlp_cart<long double>);
  bind_one_wavevector(
      "int1e_kin_cart",
      1,
      false,
      int1e_kin_cart<double>,
      int1e_kin_cart<long double>);
  bind_one_wavevector(
      "int1e_nuc_cart",
      1,
      false,
      int1e_nuc_cart<double>,
      int1e_nuc_cart<long double>);
  bind_one_wavevector(
      "int2e_cart", 3, false, int2e_cart<double>, int2e_cart<long double>);
  bind_one_wavevector(
      "int1e_ovlp_sph",
      1,
      true,
      int1e_ovlp_sph<double>,
      int1e_ovlp_sph<long double>);
  bind_one_wavevector(
      "int1e_kin_sph",
      1,
      true,
      int1e_kin_sph<double>,
      int1e_kin_sph<long double>);
  bind_one_wavevector(
      "int1e_nuc_sph",
      1,
      true,
      int1e_nuc_sph<double>,
      int1e_nuc_sph<long double>);
  bind_one_wavevector(
      "int2e_sph", 3, true, int2e_sph<double>, int2e_sph<long double>);

  const auto bind_two_wavevectors = [&](const char* name,
                                        bool spherical,
                                        TwoWaveKernel low,
                                        TwoWaveKernel high) {
    m.def(
        name,
        [=](Buffer buf,
            const Ints& shls,
            const Ints& atm,
            int natm,
            const Ints& bas,
            int nbas,
            const Reals& env,
            const Reals& k_vector,
            const Reals& l_vector,
            const std::string& precision) {
          const auto kernel = select_precision(precision, low, high);
          validate_input(buf, shls, atm, natm, bas, nbas, env, 2, spherical);
          validate_wavevector(k_vector);
          validate_wavevector(l_vector);
          kernel(
              buf.mutable_data(),
              shls.data(),
              atm.data(),
              natm,
              bas.data(),
              nbas,
              env.data(),
              k_vector.data(),
              l_vector.data());
        },
        "Compute (G_p P_k | P_l G_q), with exp(+ik.r1) exp(-il.r2). "
        "shls=[p,q]; buf has shape (dp,dq), or is flat with q fastest. "
        "Each PW includes |k|/(2*pi)^(3/2). buf must be writable C-contiguous complex128. "
        "_sph uses dp=nctr_p*(2*l_p+1), dq=nctr_q*(2*l_q+1) and includes angular normalization; "
        "_cart returns raw Cartesian components.",
        py::arg("buf").noconvert(),
        py::arg("shls"),
        py::arg("atm"),
        py::arg("natm"),
        py::arg("bas"),
        py::arg("nbas"),
        py::arg("env"),
        py::arg("k_vector"),
        py::arg("l_vector"),
        py::arg("precision") = "low");
  };
  bind_two_wavevectors(
      "int2e_gppg_cart",
      false,
      int2e_gppg_cart<double>,
      int2e_gppg_cart<long double>);
  bind_two_wavevectors(
      "int2e_gppg_sph",
      true,
      int2e_gppg_sph<double>,
      int2e_gppg_sph<long double>);
}
