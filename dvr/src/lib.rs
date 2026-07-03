use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

mod dvr;
mod grid;
mod operator;
mod result;
mod solver;

use dvr::sinc::SincDVR as RsSincDVR;
use dvr::Dvr as _;
use grid::uniform::UniformGrid as RsUniformGrid;
use grid::Grid as _;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn mat_to_vec2(mat: &faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| mat[(i, j)]).collect())
        .collect()
}

fn wfn_to_vec(mat: &faer::Mat<f64>, nroots: usize) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..nroots).map(|j| mat[(i, j)]).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// UniformGrid
// ---------------------------------------------------------------------------

#[pyclass]
struct UniformGrid {
    inner: RsUniformGrid,
}

#[pymethods]
impl UniformGrid {
    #[new]
    fn new(xmin: f64, xmax: f64, n: usize) -> Self {
        Self {
            inner: RsUniformGrid::new(xmin, xmax, n),
        }
    }

    #[getter]
    fn points(&self) -> Vec<f64> {
        self.inner.points().to_vec()
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn spacing(&self) -> f64 {
        self.inner.spacing()
    }

    #[getter]
    fn xmin(&self) -> f64 {
        self.inner.xmin()
    }

    #[getter]
    fn xmax(&self) -> f64 {
        self.inner.xmax()
    }
}

// ---------------------------------------------------------------------------
// SincDVR
// ---------------------------------------------------------------------------

#[pyclass]
struct SincDVR {
    inner: RsSincDVR<RsUniformGrid>,
}

#[pymethods]
impl SincDVR {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, mass = 1.0))]
    fn new(xmin: f64, xmax: f64, n: usize, mass: f64) -> Self {
        let grid = RsUniformGrid::new(xmin, xmax, n);
        Self {
            inner: RsSincDVR::new(grid, mass),
        }
    }

    #[getter]
    fn kinetic(&self) -> Vec<Vec<f64>> {
        mat_to_vec2(self.inner.kinetic())
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass()
    }

    /// Solve DVR eigenvalue problem.
    ///
    /// `potential`: 1-D numpy float64 array — zero-copy view into Rust.
    fn solve<'py>(&self, potential: Bound<'py, PyArray1<f64>>, nroots: usize) -> DVRResult {
        let v_slice = potential.readonly();
        let v_slice = v_slice.as_slice().expect("numpy array must be contiguous");
        let result = self.inner.solve(v_slice, nroots);
        let n = result.energies.len();
        DVRResult {
            energies: result.energies,
            wavefunctions: wfn_to_vec(&result.wavefunctions, n),
        }
    }
}

// ---------------------------------------------------------------------------
// DVRResult
// ---------------------------------------------------------------------------

#[pyclass]
struct DVRResult {
    energies: Vec<f64>,
    wavefunctions: Vec<Vec<f64>>,
}

#[pymethods]
impl DVRResult {
    #[getter]
    fn energies(&self) -> Vec<f64> {
        self.energies.clone()
    }

    #[getter]
    fn wavefunctions(&self) -> Vec<Vec<f64>> {
        self.wavefunctions.clone()
    }

    fn __repr__(&self) -> String {
        let n_states = self.energies.len();
        let n_grid = self.wavefunctions.first().map_or(0, |w| w.len());
        format!("DVRResult(n_states={n_states}, n_grid={n_grid})")
    }
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

/// Simple sanity check.
#[pyfunction]
fn hello() -> String {
    "Hello from DVR (Rust + PyO3)!".to_string()
}

#[pymodule]
fn _dvr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UniformGrid>()?;
    m.add_class::<SincDVR>()?;
    m.add_class::<DVRResult>()?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
