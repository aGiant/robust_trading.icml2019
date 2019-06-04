use rand::{
    prelude::*,
    distributions::StandardNormal,
};

pub trait PriceDynamics {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64;
}

#[derive(Debug)]
pub struct BrownianMotion {
    dt: f64,
    pub volatility: f64,
}

impl BrownianMotion {
    pub fn new(dt: f64, volatility: f64) -> BrownianMotion {
        BrownianMotion { dt, volatility, }
    }
}

impl PriceDynamics for BrownianMotion {
    fn sample_increment<R: Rng>(&self, rng: &mut R, _: f64) -> f64 {
        self.volatility * self.dt.sqrt() * rng.sample(StandardNormal)
    }
}

impl Default for BrownianMotion {
    fn default() -> BrownianMotion {
        BrownianMotion::new(0.005, 2.0)
    }
}

#[derive(Debug)]
pub struct BrownianMotionWithDrift {
    dt: f64,
    pub drift: f64,
    pub volatility: f64,
}

impl BrownianMotionWithDrift {
    pub fn new(dt: f64, drift: f64, volatility: f64) -> BrownianMotionWithDrift {
        BrownianMotionWithDrift { dt, drift, volatility, }
    }
}

impl PriceDynamics for BrownianMotionWithDrift {
    fn sample_increment<R: Rng>(&self, rng: &mut R, _: f64) -> f64 {
        self.drift * self.dt + self.volatility * self.dt.sqrt() * rng.sample(StandardNormal)
    }
}

impl Default for BrownianMotionWithDrift {
    fn default() -> BrownianMotionWithDrift {
        BrownianMotionWithDrift::new(0.005, 0.0, 2.0)
    }
}

#[derive(Debug)]
pub struct OrnsteinUhlenbeck {
    dt: f64,
    pub rate: f64,
    pub volatility: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(dt: f64, rate: f64, volatility: f64) -> OrnsteinUhlenbeck {
        OrnsteinUhlenbeck { dt, rate, volatility, }
    }
}

impl PriceDynamics for OrnsteinUhlenbeck {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64 {
        let w = BrownianMotion::new(self.dt, self.volatility);

        -self.rate * x * self.dt + w.sample_increment(rng, x)
    }
}

impl Default for OrnsteinUhlenbeck {
    fn default() -> OrnsteinUhlenbeck {
        OrnsteinUhlenbeck::new(1.0, 1.0, 1.0)
    }
}

#[derive(Debug)]
pub struct OrnsteinUhlenbeckWithDrift {
    dt: f64,
    pub rate: f64,
    pub drift: f64,
    pub volatility: f64,
}

impl OrnsteinUhlenbeckWithDrift {
    pub fn new(dt: f64, rate: f64, drift: f64, volatility: f64) -> OrnsteinUhlenbeckWithDrift {
        OrnsteinUhlenbeckWithDrift { dt, rate, drift, volatility, }
    }
}

impl PriceDynamics for OrnsteinUhlenbeckWithDrift {
    fn sample_increment<R: Rng>(&self, rng: &mut R, x: f64) -> f64 {
        let w = BrownianMotion::new(self.dt, self.volatility);

        self.rate * (self.drift - x) * self.dt + w.sample_increment(rng, x)
    }
}

impl Default for OrnsteinUhlenbeckWithDrift {
    fn default() -> OrnsteinUhlenbeckWithDrift {
        OrnsteinUhlenbeckWithDrift::new(1.0, 1.0, 0.0, 1.0)
    }
}
