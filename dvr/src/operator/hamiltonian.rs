use faer::Mat;

/// H = T + diag(V)
pub(crate) fn build_hamiltonian(kinetic: &Mat<f64>, v_diag: &[f64]) -> Mat<f64> {
    let mut h = kinetic.clone();
    for i in 0..v_diag.len() {
        h[(i, i)] += v_diag[i];
    }
    h
}
