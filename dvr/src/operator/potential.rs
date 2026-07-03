use faer::Mat;

/// Build diagonal potential energy matrix from grid-point values.
pub(crate) fn potential_matrix(v: &[f64]) -> Mat<f64> {
    let n = v.len();
    Mat::from_fn(n, n, |i, j| if i == j { v[i] } else { 0.0 })
}
