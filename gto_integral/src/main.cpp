#include "integral.hpp"

int main()
{

    /*
    >>> from pyscf import gto
    >>> mol = gto.M(atom = "H 0 0 0; H 0 0 0.7", basis = {'H': [[0,(0.48, 1.0)]]})
    >>> mol.intor("int1e_ovlp_cart")[0, 1]
    np.float64(0.657074926565322)
    >>> mol.intor("int1e_kin_cart")[0, 1]
    np.float64(0.3406411334601461)
    >>> mol.intor("int1e_nuc_cart")[0, 1]
    np.float64(-1.272771710490278)
    >>> mol.intor("int2e_cart")[0, 1, 0, 1]
    np.float64(0.33752462885462003)
    */

    std::array<double, 3> A { 0.0, 0.0, 0.0 };
    std::array<double, 3> B { 0.0, 0.0, 0.7 * 1 / 0.52917721092 };

    std::vector<double> exp1 { 0.48 };
    std::vector<double> exp2 { 0.48 };
    std::vector<double> coeff1 { 1.0 };
    std::vector<double> coeff2 { 1.0 };

    std::array<int, 3> lmn1 { 0, 0, 0 };
    std::array<int, 3> lmn2 { 0, 0, 0 };

    normalization(coeff1, lmn1, exp1);
    normalization(coeff2, lmn2, exp2);

    auto ovlp_result = overlap_elem(exp1[0], lmn1, A, exp2[0], lmn2, B) * coeff1[0] * coeff2[0];
    // fmt::print("Overlap integral: {:.10f}\n", ovlp_result);
    std::cout << std::format("Overlap integral: {:.10f}\n", ovlp_result) << std::endl;

    auto kin_result = kinetic_elem(exp1[0], lmn1, A, exp2[0], lmn2, B) * coeff1[0] * coeff2[0];
    // fmt::print("Kinetic integral: {:.10f}\n", kin_result);
    std::cout << std::format("Kinetic integral: {:.10f}\n", kin_result) << std::endl;

    auto nuc_result = -1.0 * nuclear_elem(exp1[0], lmn1, A, exp2[0], lmn2, B, A) * coeff1[0] * coeff2[0] + -1.0 * nuclear_elem(exp1[0], lmn1, A, exp2[0], lmn2, B, B) * coeff1[0] * coeff2[0];
    // fmt::print("Nuclear integral: {:.10f}\n", nuc_result);
    std::cout << std::format("Nuclear integral: {:.10f}\n", nuc_result) << std::endl;

    auto eri_result = electron_repulsion(exp1[0], lmn1, A, exp2[0], lmn2, B, exp1[0], lmn1, A, exp2[0], lmn2, B) * coeff1[0] * coeff1[0] * coeff2[0] * coeff2[0];
    // fmt::print("Electron repulsion integral: {:.10f}\n", eri_result);
    std::cout << std::format("Electron repulsion integral: {:.10f}\n", eri_result) << std::endl;
}
