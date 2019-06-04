pub trait ExecutionDynamics {
    fn match_prob(&self, offset: f64) -> f64;
}

#[derive(Debug)]
pub struct PoissonRate {
    dt: f64,
    pub scale: f64,
    pub decay: f64,
}

impl PoissonRate {
    pub fn new(dt: f64, scale: f64, decay: f64) -> PoissonRate {
        PoissonRate { dt, scale, decay, }
    }
}

impl ExecutionDynamics for PoissonRate {
    fn match_prob(&self, offset: f64) -> f64 {
        let lambda = self.scale * (-self.decay * offset).exp();

        (lambda * self.dt).max(0.0).min(1.0)
    }
}

impl Default for PoissonRate {
    fn default() -> PoissonRate {
        PoissonRate::new(0.005, 140.0, 1.5)
    }
}
