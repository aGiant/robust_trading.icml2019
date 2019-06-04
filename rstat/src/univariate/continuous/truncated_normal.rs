use crate::{
    consts::{PI_2, PI_E_2},
    core::*,
};
use rand::Rng;
use spaces::{continuous::Interval, Matrix, Vector};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct TruncatedNormal {
    pub a: f64,
    pub b: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl TruncatedNormal {
    pub fn new(a: f64, b: f64, mu: f64, sigma: f64) -> TruncatedNormal {
        assert_gt!(b > a);
        assert_bounded!(a; mu; b);
        assert_positive_real!(sigma);

        TruncatedNormal { a, b, mu, sigma, }
    }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        (x - self.mu) / self.sigma
    }
}

impl Distribution for TruncatedNormal {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (0.5 + (self.z(x) / 2.0f64.sqrt()).erf() / 2.0).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for TruncatedNormal {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        let norm = PI_2.sqrt() * self.sigma;

        (-z * z / 2.0).exp() / norm
    }
}

// impl UnivariateMoments for TruncatedNormal {
    // fn mean(&self) -> f64 {
        // self.mu
    // }

    // fn variance(&self) -> f64 {
        // self.sigma * self.sigma
    // }

    // fn skewness(&self) -> f64 {
        // 0.0
    // }

    // fn kurtosis(&self) -> f64 {
        // 0.0
    // }

    // fn excess_kurtosis(&self) -> f64 {
        // -3.0
    // }
// }

// impl Quantiles for TruncatedNormal {
    // fn quantile(&self, _: Probability) -> f64 {
        // unimplemented!()
    // }

    // fn median(&self) -> f64 {
        // self.mu
    // }
// }

// impl Modes for TruncatedNormal {
    // fn modes(&self) -> Vec<f64> {
        // vec![self.mu]
    // }
// }

// impl Entropy for TruncatedNormal {
    // fn entropy(&self) -> f64 {
        // (PI_E_2 * self.variance()).ln() / 2.0
    // }
// }

// impl FisherInformation for TruncatedNormal {
    // fn fisher_information(&self) -> Matrix {
        // let precision = self.precision();

        // unsafe {
            // Matrix::from_shape_vec_unchecked(
                // (2, 2),
                // vec![precision, 0.0, 0.0, precision * precision / 2.0],
            // )
        // }
    // }
// }
