#[derive(Clone, Copy, Default)]
pub struct WelfordRunningStats {
    pub mean: f64,
    pub count: usize,
    pub m2: f64,
}

#[allow(unused)]
impl WelfordRunningStats {
    pub fn update(&mut self, sample: f64) {
        self.count += 1;
        let old_mean = self.mean;
        self.mean += (sample - self.mean) / (self.count as f64);
        self.m2 += (sample - old_mean) * (sample - self.mean);
    }

    /// Returns the mean, or `None` if no observations have been pushed.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    /// Returns the sample variance (normalised by `n - 1`), or `None` if
    /// fewer than two observations have been pushed.
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / ((self.count - 1) as f64))
        }
    }

    /// Returns the sample standard deviation, or `None` if fewer than two
    /// observations have been pushed.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Returns the population variance (normalised by `n`), or `None` if no
    /// observations have been pushed.
    pub fn population_variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / (self.count as f64))
        }
    }

    /// Returns the population standard deviation, or `None` if no
    /// observations have been pushed.
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }
}
