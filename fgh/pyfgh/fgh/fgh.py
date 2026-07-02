"""High-level Python API for FGH vibrational calculations."""


class FGH1D:
    """1D Fourier Grid Hamiltonian solver (stub — implementation pending)."""

    def __init__(self, x_min, x_max, n_points=256, mass=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.mass = mass

    def __repr__(self):
        return f"FGH1D([{self.x_min}, {self.x_max}], n={self.n_points})"
