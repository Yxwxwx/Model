"""
TDDFT analytical nuclear gradients.
"""

from pyscf import gto, scf, dft, tddft

mol = gto.M(
    atom=[["O", 0.0, 0.0, 0], ["H", 0.0, -0.757, 0.587], ["H", 0.0, 0.757, 0.587]],
    basis="ccpvdz",
    verbose=6,
)

mf = dft.RKS(mol).x2c().set(xc="pbe0").run()
# Switch to xcfun because 3rd order GGA functional derivative is not
# available in libxc
mf._numint.libxc = dft.xcfun
postmf = tddft.TDDFT(mf).run()
from pyscf import grad

g = postmf.Gradients()
g.kernel(state=1)

# mf = scf.UHF(mol).x2c().run()
# postmf = tddft.TDHF(mf).run()
# g = postmf.nuc_grad_method()
# g.kernel()
