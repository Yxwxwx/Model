use faer::Mat;

use crate::dvr::Dvr;
use crate::grid::Grid;
use crate::operator::hamiltonian::build_hamiltonian;
use crate::operator::kinetic::kinetic_matrix;
use crate::result::SolveResult;
use crate::solver::solve_eigen;

pub struct SincDVR<G: Grid> {
    grid: G,
    mass: f64,
    kinetic: Mat<f64>,
}

impl<G: Grid> SincDVR<G> {
    pub fn new(grid: G, mass: f64) -> Self {
        let dx = grid.coordinate(1) - grid.coordinate(0); // uniform grid spacing
        let t = kinetic_matrix(grid.len(), dx);
        let scaled = if mass != 1.0 {
            t * (1.0 / mass) // or scale element-wise
        } else {
            t
        };
        Self {
            grid,
            mass,
            kinetic: scaled,
        }
    }
}

impl<G: Grid> Dvr for SincDVR<G> {
    type G = G;

    fn grid(&self) -> &Self::G {
        &self.grid
    }
    fn kinetic(&self) -> &Mat<f64> {
        &self.kinetic
    }
    fn mass(&self) -> f64 {
        self.mass
    }

    fn solve(&self, potential: &[f64], nroots: usize) -> SolveResult {
        let h = build_hamiltonian(&self.kinetic, potential);
        solve_eigen(&h, nroots)
    }
}
