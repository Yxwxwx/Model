#pragma once
// global parameters in env
// Overall cutoff for integral prescreening, value needs to be ~ln(threshold)
#define PTR_EXPCUTOFF 0
// R_C of (r-R_C) in dipole, GIAO operators
#define PTR_COMMON_ORIG 1
// R_O in 1/|r-R_O|
#define PTR_RINV_ORIG 4
// ZETA parameter for Gaussian charge distribution (Gaussian nuclear model)
#define PTR_RINV_ZETA 7
// omega parameter in range-separated coulomb operator
// LR interaction: erf(omega*r12)/r12 if omega > 0
// SR interaction: erfc(omega*r12)/r12 if omega < 0
#define PTR_RANGE_OMEGA 8
// Yukawa potential and Slater-type geminal e^{-zeta r}
#define PTR_F12_ZETA 9
// Gaussian type geminal e^{-zeta r^2}
#define PTR_GTG_ZETA 10
#define NGRIDS 11
#define PTR_GRIDS 12
#define PTR_ENV_START 20

// slots of atm
#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS 6

// slots of bas
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define KAPPA_OF 4
#define PTR_EXP 5
#define PTR_COEFF 6
#define RESERVE_BASLOT 7
#define BAS_SLOTS 8

#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]
#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]