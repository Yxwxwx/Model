use faer::Mat;

/// Colbert–Miller 1992 universal DVR kinetic energy matrix.
///
/// T_ii = (π² / 6) / dx²
/// T_{i≠j} = (-1)^{i-j} / ((i-j)² * dx²)
pub(crate) fn kinetic_matrix(n: usize, dx: f64) -> Mat<f64> {
    let dx2_inv = 1.0 / (dx * dx);
    let diag = dx2_inv * std::f64::consts::TAU.powi(2) / 24.0; // π²/6 = τ²/24

    Mat::from_fn(n, n, |i, j| {
        if i == j {
            diag
        } else {
            let diff = i as isize - j as isize;
            let d2 = (diff as f64) * (diff as f64);
            let sign = if diff % 2 == 0 { 1.0 } else { -1.0 };
            dx2_inv * sign / d2
        }
    })
}
