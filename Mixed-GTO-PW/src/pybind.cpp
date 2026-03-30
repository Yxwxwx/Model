#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "integral.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libMixedInt, m) {
  m.doc() = "GPW Integration Library for PySCF";

  m.def(
      "int1e_ovlp_cart",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             buf,
         py::array_t<int> shls, py::array_t<int> atm, int natm,
         py::array_t<int> bas, int nbas, py::array_t<double> env,
         py::array_t<double> k_vector) {
        // 1. all arrays to raw pointers
        auto buf_ptr = static_cast<std::complex<double>*>(buf.mutable_data());
        auto shls_ptr = static_cast<const int*>(shls.data());
        auto atm_ptr = static_cast<const int*>(atm.data());
        auto bas_ptr = static_cast<const int*>(bas.data());
        auto env_ptr = static_cast<const double*>(env.data());
        auto k_ptr = static_cast<const double*>(k_vector.data());

        // 2. C++ core function
        int1e_ovlp_cart(buf_ptr, shls_ptr, atm_ptr, natm, bas_ptr, nbas,
                        env_ptr, k_ptr);
      },
      "Compute 1e overlap integrals (GPW) in Cartesian coordinates",
      py::arg("buf"), py::arg("shls"), py::arg("atm"), py::arg("natm"),
      py::arg("bas"), py::arg("nbas"), py::arg("env"), py::arg("k_vector"));

  m.def(
      "int1e_kin_cart",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             buf,
         py::array_t<int> shls, py::array_t<int> atm, int natm,
         py::array_t<int> bas, int nbas, py::array_t<double> env,
         py::array_t<double> k_vector) {
        // 1. 获取所有数组的原始指针
        auto buf_ptr = static_cast<std::complex<double>*>(buf.mutable_data());
        auto shls_ptr = static_cast<const int*>(shls.data());
        auto atm_ptr = static_cast<const int*>(atm.data());
        auto bas_ptr = static_cast<const int*>(bas.data());
        auto env_ptr = static_cast<const double*>(env.data());
        auto k_ptr = static_cast<const double*>(k_vector.data());

        // 2. C++ core function
        int1e_kin_cart(buf_ptr, shls_ptr, atm_ptr, natm, bas_ptr, nbas, env_ptr,
                       k_ptr);
      },
      "Compute 1e kinetic integrals (GPW) in Cartesian coordinates",
      py::arg("buf"), py::arg("shls"), py::arg("atm"), py::arg("natm"),
      py::arg("bas"), py::arg("nbas"), py::arg("env"), py::arg("k_vector"));

  m.def(
      "int1e_nuc_cart",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             buf,
         py::array_t<int> shls, py::array_t<int> atm, int natm,
         py::array_t<int> bas, int nbas, py::array_t<double> env,
         py::array_t<double> k_vector) {
        // 1. all arrays to raw pointers
        auto buf_ptr = static_cast<std::complex<double>*>(buf.mutable_data());
        auto shls_ptr = static_cast<const int*>(shls.data());
        auto atm_ptr = static_cast<const int*>(atm.data());
        auto bas_ptr = static_cast<const int*>(bas.data());
        auto env_ptr = static_cast<const double*>(env.data());
        auto k_ptr = static_cast<const double*>(k_vector.data());

        // 2. C++ core function
        int1e_nuc_cart(buf_ptr, shls_ptr, atm_ptr, natm, bas_ptr, nbas, env_ptr,
                       k_ptr);
      },
      "Compute 1e nuclear integrals (GPW) in Cartesian coordinates",
      py::arg("buf"), py::arg("shls"), py::arg("atm"), py::arg("natm"),
      py::arg("bas"), py::arg("nbas"), py::arg("env"), py::arg("k_vector"));

  m.def(
      "int2e_cart",
      [](py::array_t<std::complex<double>,
                     py::array::c_style | py::array::forcecast>
             buf,
         py::array_t<int> shls, py::array_t<int> atm, int natm,
         py::array_t<int> bas, int nbas, py::array_t<double> env,
         py::array_t<double> k_vector) {
        // 1. all arrays to raw pointers
        auto buf_ptr = static_cast<std::complex<double>*>(buf.mutable_data());
        auto shls_ptr = static_cast<const int*>(shls.data());
        auto atm_ptr = static_cast<const int*>(atm.data());
        auto bas_ptr = static_cast<const int*>(bas.data());
        auto env_ptr = static_cast<const double*>(env.data());
        auto k_ptr = static_cast<const double*>(k_vector.data());

        // 2. C++ core function
        int2e_cart(buf_ptr, shls_ptr, atm_ptr, natm, bas_ptr, nbas, env_ptr,
                   k_ptr);
      },
      "Compute 2e electron electron integrals (GPW) in Cartesian coordinates",
      py::arg("buf"), py::arg("shls"), py::arg("atm"), py::arg("natm"),
      py::arg("bas"), py::arg("nbas"), py::arg("env"), py::arg("k_vector"));
}