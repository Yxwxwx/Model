use faer::Side;

use crate::result::SolveResult;

pub(crate) fn solve_eigen(h: &faer::Mat<f64>, nroots: usize) -> SolveResult {
    let n = h.nrows();
    let nroots = nroots.min(n); // can't return more states than grid points
    let evd = h
        .self_adjoint_eigen(Side::Lower)
        .expect("Diagonalization failed");
    let eigenvalues_diag = evd.S();
    let energies: Vec<f64> = (0..nroots).map(|i| eigenvalues_diag[i]).collect();

    let full_wfns = evd.U();
    let wfns = full_wfns.submatrix(0, 0, n, nroots).to_owned();

    SolveResult {
        energies,
        wavefunctions: wfns,
    }
}
