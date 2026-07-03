use faer::Mat;

/// Eigenvalue solution container.
pub struct SolveResult {
    pub energies: Vec<f64>,
    pub wavefunctions: Mat<f64>, // (n_grid, nroots)
}
