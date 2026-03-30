from pyscf import gto, mcscf, fci
from pyscf.tools import fcidump

mol = gto.M(atom="n 0 0 -0.55; n 0 0 0.55", basis="sto-3g")
myhf = mol.RHF().run()

cisolver = fci.FCI(myhf)
print("E(FCI) = %.12f" % cisolver.kernel()[0])

# ncas, nelecas = (10, 14)
# mycas = mcscf.CASCI(myhf, ncas, nelecas)
# mycas.kernel()
# fcidump.from_mcscf(mycas, "fcidump.example1")
