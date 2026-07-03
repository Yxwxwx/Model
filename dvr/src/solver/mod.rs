mod dense;
pub(crate) use dense::solve_eigen;

pub(crate) trait EigenSolver {
    fn solve(&self, h: &faer::Mat<f64>, nroots: usize) -> crate::result::SolveResult;
}
