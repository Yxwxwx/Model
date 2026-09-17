import numpy as np
import quimb.tensor as qtn
from tqdm.auto import tqdm

from tddmrg.tdvp import TDVP


L = 10

psi0 = qtn.MPS_computational_state("0101010101")
H = qtn.MPO_ham_heis(L)

tdvp = TDVP(
    psi0,
    H,
    dt=0.01,
    order=1,
    split_opts={
        "max_bond": 64,
        "cutoff": 1e-12,
    },
)

ts = np.linspace(0.0, 1.0, 11)
records = []

for psit in tdvp.at_times(ts):
    records.append(
        (
            tdvp.t,
            psit.max_bond(),
            tdvp.last_trunc_err,
        )
    )

print()
print(f"{'time':>10} {'max_bond':>10} {'trunc_error':>16}")
print("-" * 40)

for t, chi, trunc in records:
    print(f"{t:10.3f} {chi:10d} {trunc:16.6e}")
