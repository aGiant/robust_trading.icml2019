use crate::{
    consts::PI_2,
    core::*,
    univariate::continuous::Normal,
};
use rand::Rng;
use spaces::{continuous::PositiveReals, Matrix};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct LogNormal(Normal);

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> LogNormal {
        LogNormal(Normal::new(mu, sigma))
    }

    fn z(&self, x: f64) -> f64 {
        self.0.z(x.ln())
    }
}

impl Default for LogNormal {
    fn default() -> LogNormal {
        LogNormal(Normal::default())
    }
}

impl Into<rand::distributions::LogNormal> for LogNormal {
    fn into(self) -> rand::distributions::LogNormal {
        rand::distributions::LogNormal::new(self.0.mu, self.0.sigma)
    }
}

impl Into<rand::distributions::LogNormal> for &LogNormal {
    fn into(self) -> rand::distributions::LogNormal {
        rand::distributions::LogNormal::new(self.0.mu, self.0.sigma)
    }
}

impl Distribution for LogNormal {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn cdf(&self, x: f64) -> Probability { self.0.cdf(x.ln()) }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distributions::{LogNormal as LNSampler, Distribution as DistSampler};

        let sampler: LNSampler = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for LogNormal {
    fn pdf(&self, x: f64) -> f64 { self.0.pdf(x.ln()) }
}

impl UnivariateMoments for LogNormal {
    fn mean(&self) -> f64 {
        (self.0.mu + self.0.sigma * self.0.sigma / 2.0).exp()
    }

    fn variance(&self) -> f64 {
        let sigma2 = self.0.sigma * self.0.sigma;

        (sigma2.exp() - 1.0) * (2.0 * self.0.mu + sigma2).exp()
    }

    fn skewness(&self) -> f64 {
        let sigma2 = self.0.sigma * self.0.sigma;

        (sigma2.exp() + 2.0) * (sigma2.exp() - 1.0).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let sigma2 = self.0.sigma * self.0.sigma;

        (4.0 * sigma2).exp() + 2.0 * (3.0 * sigma2).exp() + 3.0 * (2.0 * sigma2).exp() - 6.0
    }
}

impl Quantiles for LogNormal {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.0.mu.exp()
    }
}

impl Modes for LogNormal {
    fn modes(&self) -> Vec<f64> {
        vec![(self.0.mu - self.0.sigma * self.0.sigma).exp()]
    }
}

impl Entropy for LogNormal {
    fn entropy(&self) -> f64 {
        (self.0.sigma * (self.0.mu + 0.5).exp() * PI_2.sqrt()).log2()
    }
}

impl FisherInformation for LogNormal {
    fn fisher_information(&self) -> Matrix {
        let one_over_sigma2 = 1.0 / self.0.sigma / self.0.sigma;

        unsafe {
            Matrix::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    one_over_sigma2,
                    0.0,
                    0.0,
                    one_over_sigma2 * one_over_sigma2 / 2.0,
                ],
            )
        }
    }
}

impl fmt::Display for LogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lognormal({}, {})", self.0.mu, self.variance())
    }
}
