use crate::{
    consts::{ONE_OVER_PI, PI},
    core::*,
};
use rand::Rng;
use spaces::continuous::Reals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Cauchy {
    pub x0: f64,
    pub gamma: f64,
}

impl Cauchy {
    pub fn new(x0: f64, gamma: f64) -> Cauchy {
        Cauchy { x0, gamma }
    }

    pub fn fwhm(&self) -> f64 {
        2.0 * self.gamma
    }

    #[inline(always)]
    fn z(&self, x: f64) -> f64 {
        (x - self.x0) / self.gamma
    }
}

impl Default for Cauchy {
    fn default() -> Cauchy {
        Cauchy {
            x0: 0.0,
            gamma: 1.0,
        }
    }
}

impl Into<rand::distributions::Cauchy> for Cauchy {
    fn into(self) -> rand::distributions::Cauchy {
        rand::distributions::Cauchy::new(self.x0, self.gamma)
    }
}

impl Into<rand::distributions::Cauchy> for &Cauchy {
    fn into(self) -> rand::distributions::Cauchy {
        rand::distributions::Cauchy::new(self.x0, self.gamma)
    }
}

impl Distribution for Cauchy {
    type Support = Reals;

    fn support(&self) -> Reals {
        Reals
    }

    fn cdf(&self, x: f64) -> Probability {
        (ONE_OVER_PI * self.z(x).atan() + 0.5).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distributions::{Cauchy as CauchySampler, Distribution as DistSampler};

        let sampler: CauchySampler = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);

        1.0 / PI / self.gamma / (1.0 + z * z)
    }
}

impl Quantiles for Cauchy {
    fn quantile(&self, p: Probability) -> f64 {
        self.x0 + self.gamma * (PI * (f64::from(p) - 0.5)).tan()
    }

    fn median(&self) -> f64 {
        self.x0
    }
}

impl Modes for Cauchy {
    fn modes(&self) -> Vec<f64> {
        vec![self.x0]
    }
}

impl Entropy for Cauchy {
    fn entropy(&self) -> f64 {
        (4.0 * PI * self.gamma).ln()
    }
}

impl Convolution<Cauchy> for Cauchy {
    fn convolve(self, rv: Cauchy) -> ConvolutionResult<Cauchy> {
        Self::convolve_pair(self, rv)
    }

    fn convolve_pair(a: Cauchy, b: Cauchy) -> ConvolutionResult<Cauchy> {
        Ok(Cauchy::new(a.x0 + b.x0, a.gamma + b.gamma))
    }
}

impl fmt::Display for Cauchy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cauchy({}, {})", self.x0, self.gamma)
    }
}
