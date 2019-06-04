use crate::core::*;
use rand;
use spaces::{Vector, Matrix, discrete::Binary};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    pub p: Probability,
    q: Probability,

    variance: f64,
}

impl Bernoulli {
    pub fn new<P: Into<Probability>>(p: P) -> Bernoulli {
        let p = p.into();

        Bernoulli {
            p: p,
            q: !p,

            variance: (p * !p).into(),
        }
    }
}

impl Into<rand::distributions::Bernoulli> for Bernoulli {
    fn into(self) -> rand::distributions::Bernoulli {
        rand::distributions::Bernoulli::new(f64::from(self.p))
    }
}

impl Into<rand::distributions::Bernoulli> for &Bernoulli {
    fn into(self) -> rand::distributions::Bernoulli {
        rand::distributions::Bernoulli::new(f64::from(self.p))
    }
}

impl Distribution for Bernoulli {
    type Support = Binary;

    fn support(&self) -> Binary { Binary }

    fn cdf(&self, k: bool) -> Probability {
        if k {
            1.0.into()
        } else {
            0.0.into()
        }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen_bool(self.p.into())
    }
}

impl DiscreteDistribution for Bernoulli {
    fn pmf(&self, k: bool) -> Probability {
        match k {
            true => self.p,
            false => self.q,
        }
    }
}

impl UnivariateMoments for Bernoulli {
    fn mean(&self) -> f64 { self.p.into() }

    fn variance(&self) -> f64 {
        self.variance
    }

    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * f64::from(self.p)) / self.variance.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        1.0 / self.variance - 6.0
    }

    fn excess_kurtosis(&self) -> f64 {
        1.0 / self.variance - 9.0
    }
}

impl Quantiles for Bernoulli {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        match f64::from(self.p) {
            p if (p - 0.5).abs() < 1e-7 => 0.5,
            p if (p < 0.5) => 0.0,
            _ => 1.0,
        }
    }
}

impl Modes for Bernoulli {
    fn modes(&self) -> Vec<bool> {
        use std::cmp::Ordering::*;

        match self.p.partial_cmp(&self.q) {
            Some(Less) => vec![false],
            Some(Equal) => vec![false, true],
            Some(Greater) => vec![false],
            None => unreachable!(),
        }
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        let p: f64 = self.p.into();
        let q: f64 = self.q.into();

        if q.abs() < 1e-7 || (q - 1.0).abs() < 1e-7 {
            0.0
        } else {
            -q * q.ln() - p*p.ln()
        }
    }
}

impl FisherInformation for Bernoulli {
    fn fisher_information(&self) -> Matrix {
        Matrix::from_elem((1, 1), 1.0 / self.variance)
    }
}

impl MLE for Bernoulli {
    fn fit_mle(samples: Vector<bool>) -> Self {
        let n = samples.len() as f64;

        Bernoulli::new(samples.fold(0, |acc, v| acc + *v as u64) as f64 / n)
    }
}

impl fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ber({})", self.p)
    }
}
