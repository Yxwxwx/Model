pub mod uniform;

/// One-dimensional grid
pub trait Grid {
    fn points(&self) -> &[f64];
    fn len(&self) -> usize;
    fn coordinate(&self, i: usize) -> f64;
}
