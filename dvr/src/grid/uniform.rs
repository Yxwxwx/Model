use super::Grid;

pub struct UniformGrid {
    xmin: f64,
    xmax: f64,
    n: usize,
    dx: f64,
    points: Vec<f64>,
}

impl UniformGrid {
    pub fn new(xmin: f64, xmax: f64, n: usize) -> Self {
        let dx = (xmax - xmin) / (n - 1) as f64;
        let points: Vec<f64> = (0..n).map(|i| xmin + i as f64 * dx).collect();
        Self {
            xmin,
            xmax,
            n,
            dx,
            points,
        }
    }

    pub fn spacing(&self) -> f64 {
        self.dx
    }
    pub fn xmin(&self) -> f64 {
        self.xmin
    }
    pub fn xmax(&self) -> f64 {
        self.xmax
    }
}

impl Grid for UniformGrid {
    fn points(&self) -> &[f64] {
        &self.points
    }
    fn len(&self) -> usize {
        self.n
    }
    fn coordinate(&self, i: usize) -> f64 {
        self.points[i]
    }
}
