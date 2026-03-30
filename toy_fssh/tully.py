import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt

QUIET = True


@dataclass
class Neuclus:
    mass: float
    x: float
    v: float
    state: int


@dataclass
class DensityMatrix:
    rho11: complex
    rho22: complex
    rho12: complex


class Tully:
    def __init__(self):
        self.A, self.B = 0.01, 1.6
        self.C, self.D = 0.005, 1.0

    def compute_H(self, x: float) -> np.ndarray:
        V11 = (
            self.A * (1 - np.exp(-self.B * x))
            if x > 0
            else -self.A * (1 - np.exp(self.B * x))
        )
        V12 = self.C * np.exp(-self.D * x**2)
        H = np.array([[V11, V12], [V12, -V11]], dtype=np.complex128)
        return H

    def get_adiabatic_state(
        self, x: float
    ) -> Tuple[float, np.ndarray, float, np.ndarray]:
        H = self.compute_H(x)
        E, phi = np.linalg.eigh(H)
        return float(E[0]), phi[:, 0], float(E[1]), phi[:, 1]

    def get_derivative_t(self, nuc: Neuclus, den: DensityMatrix) -> np.ndarray:
        E1, phi1, E2, phi2 = self.get_adiabatic_state(nuc.x)

        dx = 1e-5
        dH = (self.compute_H(nuc.x + dx) - self.compute_H(nuc.x)) / dx

        F1 = phi1.T @ dH @ phi1
        F2 = phi2.T @ dH @ phi2
        d12 = (phi1.T @ dH @ phi2) / (E2 - E1)

        du = np.zeros(5, dtype=np.complex128)
        du[0] = nuc.v
        du[1] = -F1 / nuc.mass if nuc.state == 1 else -F2 / nuc.mass
        du[2] = -2.0 * (den.rho12.conjugate() * nuc.v * d12).real
        du[3] = 2.0 * (den.rho12 * nuc.v * d12.conjugate()).real
        du[4] = (
            -(0 + 1j) * den.rho12 * (E1 - E2)
            + den.rho11 * nuc.v * d12
            - den.rho22 * nuc.v * d12
        )
        return du

    def RK4(
        self, dt: float, nuc: Neuclus, den: DensityMatrix
    ) -> Tuple[Neuclus, DensityMatrix]:
        def get_state(
            nuc_old: Neuclus,
            den_old: DensityMatrix,
            k: np.ndarray,
            dt: float,
            step: float = 1.0,
        ):
            nuc_new = Neuclus(
                nuc_old.mass,
                nuc_old.x + step * k[0].real * dt,
                nuc_old.v + step * k[1].real * dt,
                nuc_old.state,
            )
            den_new = DensityMatrix(
                den_old.rho11 + step * k[2] * dt,
                den_old.rho22 + step * k[3] * dt,
                den_old.rho12 + step * k[4] * dt,
            )
            return nuc_new, den_new

        k1 = self.get_derivative_t(nuc, den)
        nuc2, den2 = get_state(nuc, den, k1, dt, 0.5)
        k2 = self.get_derivative_t(nuc2, den2)
        nuc3, den3 = get_state(nuc2, den2, k2, dt, 0.5)
        k3 = self.get_derivative_t(nuc3, den3)
        nuc4, den4 = get_state(nuc3, den3, k3, dt)
        k4 = self.get_derivative_t(nuc4, den4)

        k_tot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return get_state(nuc, den, k_tot, dt)

    def test_hop(self, dt: float, nuc: Neuclus, den: DensityMatrix) -> bool:
        du = self.get_derivative_t(nuc, den)
        b12, b21 = du[2].real, du[3].real

        E1, _, E2, _ = self.get_adiabatic_state(nuc.x)
        random_val = np.random.random()
        if nuc.state == 1:
            prob = dt * b21 / den.rho11.real
            energy_check = E1 + nuc.mass * nuc.v**2 / 2.0 - E2 >= 0

            hop = prob >= random_val and energy_check

            if hop and not QUIET:
                print(
                    f"Hop attempt from state 1->2: prob={prob:.4f}, random={random_val:.4f}, energy_check={energy_check}"
                )
            return hop

        prob = dt * b12 / den.rho22.real
        hop = prob >= random_val
        if hop and not QUIET:
            print(
                f"Hop attempt from state 2->1: prob={prob:.4f}, random={random_val:.4f}"
            )
        return hop

    def step_upgrade(
        self, dt: float, nuc: Neuclus, den: DensityMatrix
    ) -> Tuple[Neuclus, DensityMatrix]:
        E1, _, E2, _ = self.get_adiabatic_state(nuc.x)
        nuc_new, den_new = self.RK4(dt, nuc, den)

        if self.test_hop(dt, nuc_new, den_new):
            old_state = nuc_new.state
            old_v = nuc_new.v

            if nuc_new.state == 1:
                nuc_new.state = 2
                v = np.sqrt(
                    2.0 * (E1 + nuc_new.mass * nuc_new.v**2 / 2.0 - E2) / nuc_new.mass
                )
            else:
                nuc_new.state = 1
                v = np.sqrt(
                    2.0 * (E2 + nuc_new.mass * nuc_new.v**2 / 2.0 - E1) / nuc_new.mass
                )
            nuc_new.v = v if nuc_new.v > 0 else -v

            if not QUIET:
                print(
                    f"Hop successful: state {old_state}->{nuc_new.state}, velocity {old_v:.4f}->{nuc_new.v:.4f}"
                )

        return nuc_new, den_new

    def simulate(
        self, x0: float, v0: float, dt: float = 0.5, Ntraj: int = 1000
    ) -> Tuple[int, int, int]:
        counts = np.zeros(3, dtype=int)  # [reflex, hop, transm]
        # print(f"\nStarting simulation with x0={x0}, v0={v0}, dt={dt}, Ntraj={Ntraj}")

        for traj in range(Ntraj):
            nuc = Neuclus(2000, x0, v0, 1)
            den = DensityMatrix(1 + 0j, 0 + 0j, 0 + 0j)

            while -10 <= nuc.x <= 10:
                nuc, den = self.step_upgrade(dt, nuc, den)

            if nuc.state == 2:
                counts[1] += 1  # hop
            elif nuc.x < 0:
                counts[0] += 1  # reflex
            else:
                counts[2] += 1  # transm

        print(
            f"Simulation complete: Reflection={counts[0]}, Hop={counts[1]}, Transmission={counts[2]}"
        )
        return tuple(counts)


def main():
    from mpi4py import MPI

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define momentum array
    pspan = np.array(
        [
            1.42,
            4.12,
            4.539,
            4.550,
            5.550,
            5.687,
            7.566,
            7.811,
            8.389,
            9.38389,
            9.810,
            11.115,
            12.940,
            14.360,
            15.924,
            19.621,
            22.607,
            25.308,
            27.726,
            30.0,
        ]
    )

    # Calculate local workload
    counts = [
        len(pspan) // size + (1 if i < len(pspan) % size else 0) for i in range(size)
    ]
    displacements = [sum(counts[:i]) for i in range(size)]
    local_pspan = pspan[displacements[rank] : displacements[rank] + counts[rank]]

    # Initialize local results array
    local_results = np.zeros((len(local_pspan), 4))  # Added column for momentum

    # Perform local calculations
    model = Tully()
    for i, p in enumerate(local_pspan):
        reflex, hop, transm = model.simulate(x0=-10.0, v0=p / 2000)
        local_results[i] = [p, reflex, hop, transm]  # Store momentum with results

    # Gather results from all processes
    if rank == 0:
        global_results = np.empty((len(pspan), 4), dtype=np.float64)
    else:
        global_results = None

    # Gather results using MPI
    comm.Gatherv(
        sendbuf=local_results,
        recvbuf=(
            global_results,
            [c * 4 for c in counts],
            [d * 4 for d in displacements],
            MPI.DOUBLE,
        ),
        root=0,
    )

    # Process 0 handles data saving and plotting
    if rank == 0:
        # Sort results by momentum
        global_results = global_results[global_results[:, 0].argsort()]

        # Create combined plot
        plt.figure(figsize=(10, 6))
        labels = ["Reflection", "Hop", "Transmission"]
        colors = ["b", "g", "r"]
        markers = ["o", "^", "s"]

        for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
            plt.plot(
                global_results[:, 0],
                global_results[:, i + 1],
                color=color,
                marker=marker,
                label=label,
            )

        plt.xlabel("Momentum (a.u.)")
        plt.ylabel("Probability")
        plt.title("Tully Model: Quantum Dynamics Results")
        plt.legend()
        plt.grid(True)
        plt.savefig("tully_combined_results.png")
        plt.close()


def test():
    xlist = []
    elist = []
    nuc = Neuclus(mass=2000, x=-10, v=25 / 2000, state=1)
    den = DensityMatrix(rho11=1 + 0j, rho22=0 + 0j, rho12=0 + 0j)
    model = Tully()

    while -10.0 <= nuc.x <= 10.0:
        nuc, den = model.step_upgrade(dt=0.5, nuc=nuc, den=den)
        xlist.append(nuc.x)

        # 根据当前所在态，选择对应的基态或激发态势能
        E1, _, E2, _ = model.get_adiabatic_state(x=nuc.x)
        e = E1 if nuc.state == 1 else E2
        elist.append(e)

    fig, ax = plt.subplots()
    ax.plot(xlist, elist, label="state")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Energy (a.u.)")
    ax.set_title("Trajectory on Adiabatic Potential Surface")
    ax.legend()
    # plt.show()


if __name__ == "__main__":
    # test()
    #
    main()

    # nuc = Neuclus(mass=2000, x=2, v=30 / 2000, state=1)
    # den = DensityMatrix(rho11=1 + 0j, rho22=0 + 0j, rho12=0 + 0j)
    # model = Tully()
    # print(model.get_derivative_t(nuc, den))
