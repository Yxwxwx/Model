pub mod sinc;

use crate::grid::Grid;
use crate::result::SolveResult;
use faer::Mat;

pub trait Dvr {
    type G: Grid;

    fn grid(&self) -> &Self::G;
    fn kinetic(&self) -> &Mat<f64>;
    fn mass(&self) -> f64;
    fn solve(&self, potential: &[f64], nroots: usize) -> SolveResult;
}
