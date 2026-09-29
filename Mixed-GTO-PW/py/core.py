import libMixedInt as lib
from pyscf import gto
import numpy as np

# Set to "high" to use long-double arithmetic.
precision = "low"

mol = gto.M(atom="Ne 0 0 0", basis="unc-cc-pvdz")

atm = mol._atm
natm = mol.natm
bas = mol._bas
nbas = mol.nbas
env = mol._env
nao = mol.nao_nr()
ao_loc = mol.ao_loc_nr()
ks = [np.array([1.0, 0.0, 0.0], dtype=np.float64)]
# Spherical kernels include angular normalization in C++.

print("atm : \n", atm)
print("natm : ", natm)
print("bas : \n", bas)
print("nbas : ", nbas)
print("env : \n", env)

T = np.zeros((nao, 1), dtype=np.complex128)
V = np.zeros((nao, 1), dtype=np.complex128)
for i in range(nbas):
    i0, i1 = ao_loc[i], ao_loc[i + 1]
    di = i1 - i0
    shls = np.array([i])
    for ik, k in enumerate(ks):
        buf = np.zeros((di, 1), dtype=np.complex128)
        lib.int1e_kin_sph(buf, shls, atm, natm, bas, nbas, env, k, precision=precision)
        T[i0:i1, ik] = buf[:, 0]

for i in range(nbas):
    i0, i1 = ao_loc[i], ao_loc[i + 1]
    di = i1 - i0
    shls = np.array([i])
    for ik, k in enumerate(ks):
        buf = np.zeros((di, 1), dtype=np.complex128)
        lib.int1e_nuc_sph(buf, shls, atm, natm, bas, nbas, env, k, precision=precision)
        V[i0:i1, ik] = buf[:, 0]

h1e = T + V
# I_pkqr_sph (p, k, q, r)
I_pkqr_sph = np.zeros((nao, len(ks), nao, nao), dtype=np.complex128)
# I_rpkq_sph (r, p, k, q)
I_rpkq_sph = np.zeros((nao, nao, len(ks), nao), dtype=np.complex128)

import time

start = time.time()
for p in range(nbas):
    p0, p1 = ao_loc[p], ao_loc[p + 1]
    dp = p1 - p0

    for q in range(nbas):
        q0, q1 = ao_loc[q], ao_loc[q + 1]
        dq = q1 - q0

        for r in range(nbas):
            r0, r1 = ao_loc[r], ao_loc[r + 1]
            dr = r1 - r0

            # --- 1. (p, k | q, r) ---
            shls = np.array([p, q, r], dtype=np.int32)
            buf = np.empty(dp * dq * dr, dtype=np.complex128)

            for ik, k in enumerate(ks):
                lib.int2e_sph(buf, shls, atm, natm, bas, nbas, env, k, precision=precision)
                I_pkqr_sph[p0:p1, ik, q0:q1, r0:r1] = buf.reshape(dp, dq, dr)

            # --- 2. (k, q | r, p) <==> (r, p | k, q) ---
            shls = np.array([q, r, p], dtype=np.int32)
            for ik, k in enumerate(ks):
                lib.int2e_sph(buf, shls, atm, natm, bas, nbas, env, k, precision=precision)
                I_rpkq_sph[r0:r1, p0:p1, ik, q0:q1] = buf.reshape(dq, dr, dp).transpose(
                    1, 2, 0
                )

print("time : ", time.time() - start)
