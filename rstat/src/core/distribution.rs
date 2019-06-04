use crate::core::Probability;
use ndarray::{Array, Dimension, ShapeBuilder};
use rand::Rng;
use spaces::{Space, Vector, product::{DoubleSpace, LinearSpace}};

pub struct Sampler<D, R> {
    pub(super) distribution: D,
    pub(super) rng: R,
}

impl<D, R> Iterator for Sampler<D, R>
    where D: Distribution,
          R: Rng,
{
    type Item = <D::Support as Space>::Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.distribution.sample(&mut self.rng))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
}

macro_rules! ln_variant {
    ($(#[$attr:meta])* => $name:ident, $name_ln:ident, $x:ty) => {
        $(#[$attr])*
        fn $name_ln(&self, x: $x) -> f64 {
            f64::from(self.$name(x)).ln()
        }
    }
}

macro_rules! batch_variant {
    ($(#[$attr:meta])* => $name:ident, $name_batch:ident, $x:ty, $res:ty) => {
        $(#[$attr])*
        fn $name_batch(&self, xs: Vector<$x>) -> Vector<$res> {
            xs.mapv(|x| self.$name(x))
        }
    }
}

pub type UnivariateDistribution<S> = Distribution<Support = S>;
pub type BivariateDistribution<S> = Distribution<Support = DoubleSpace<S>>;
pub type MultivariateDistribution<S> = Distribution<Support = LinearSpace<S>>;

pub trait Distribution {
    type Support: Space;

    /// Returns an instance of the support `Space`, `Self::Support`.
    fn support(&self) -> Self::Support;

    /// Evaluates the cumulative distribution function (CDF) at `x`.
    ///
    /// The CDF is defined as the probability that a random variable X takes on a value less than
    /// or equal to `x`: `F(x) = P(X <= x)`.
    fn cdf(&self, x: <Self::Support as Space>::Value) -> Probability {
        unimplemented!()
    }

    /// Evaluates the complementary cumulative distribution function at `x`.
    ///
    /// The complementary CDF is defined as the probability that a random variable X takes on a
    /// value strictly greater than `x`: `P(X > x) = 1 - F(x)`, where `F(.)` is the CDF.
    fn ccdf(&self, x: <Self::Support as Space>::Value) -> Probability {
        !self.cdf(x)
    }

    ln_variant!(
        /// Evaluates the log CDF at `x`: `ln F(x)`.
        => cdf, logcdf, <Self::Support as Space>::Value
    );

    ln_variant!(
        /// Evaluates the log complementary CDF at `x`: `ln (1 - F(x))`.
        => ccdf, logccdf, <Self::Support as Space>::Value
    );

    batch_variant!(
        /// Evaluates the CDF element-wise for a batch `xs`.
        => cdf, cdf_batch, <Self::Support as Space>::Value, Probability
    );

    batch_variant!(
        /// Evaluates the complementary CDF element-wise for a batch `xs`.
        => ccdf, ccdf_batch, <Self::Support as Space>::Value, Probability
    );

    batch_variant!(
        /// Evaluates the log CDF element-wise for a batch `xs`.
        => logcdf, logcdf_batch, <Self::Support as Space>::Value, f64
    );

    batch_variant!(
        /// Evaluates the log complementary CDF element-wise for a batch `xs`.
        => logccdf, logccdf_batch, <Self::Support as Space>::Value, f64
    );

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self::Support as Space>::Value;

    fn sample_n<D, Sh, R>(
        &self,
        rng: &mut R,
        shape: Sh
    ) -> Array<<Self::Support as Space>::Value, D>
        where D: Dimension,
              Sh: ShapeBuilder<Dim=D>,
              R: Rng + ?Sized,
    {
        Array::from_shape_fn(shape, move |_| self.sample(rng))
    }

    fn sample_iter<R>(self, rng: R) -> Sampler<Self, R>
        where Self: Sized,
              R: Rng,
    {
        Sampler {
            distribution: self,
            rng: rng,
        }
    }
}

pub trait DiscreteDistribution: Distribution {
    /// Evaluates the probability mass function (PMF) at `x`.
    ///
    /// The PMF is defined as the probability that a random variable `X` takes a value exactly
    /// equal to `x`: `f(x) = P(X = x) = P({s in S : X(s) = x})`. We require that all sum of
    /// probabilities over all possible outcomes sums to 1.
    fn pmf(&self, x: <Self::Support as Space>::Value) -> Probability {
        self.logpmf(x).exp().into()
    }

    ln_variant!(
        /// Evaluates the log PMF at `x`.
        => pmf, logpmf, <Self::Support as Space>::Value
    );

    batch_variant!(
        /// Evaluates the PMF element-wise for a batch `xs`.
        => pmf, pmf_batch, <Self::Support as Space>::Value, Probability
    );

    batch_variant!(
        /// Evaluates the log PMF element-wise for a batch `xs`.
        => logpmf, logpmf_batch, <Self::Support as Space>::Value, f64
    );
}

pub trait ContinuousDistribution: Distribution {
    /// Evaluates the probability density function (PDF) at `x`.
    ///
    /// The PDF can be interpreted as the relative likelihood that a random variable X takes on a
    /// value equal to `x`. For absolutely continuous univariate distributions it is defined by the
    /// derivative of the CDF: `f(x) = F'(x)`. Intuitively, one may think of `f(x)dx` that as
    /// representing the probability that the random variable `X` lies in the infinitesimal
    /// interval `[x, x + dx]`.
    ///
    /// Alternatively, one may interpret the PDF, for infinitely small `dt`, as the following:
    /// `f(t)dt = P(t < X < t + dt)`.
    fn pdf(&self, x: <Self::Support as Space>::Value) -> f64 {
        self.logpdf(x).exp().into()
    }

    ln_variant!(
        /// Evaluates the log PDF at `x`.
        => pdf, logpdf, <Self::Support as Space>::Value
    );

    batch_variant!(
        /// Evaluates the PDF element-wise for a batch `xs`.
        => pdf, pdf_batch, <Self::Support as Space>::Value, f64
    );

    batch_variant!(
        /// Evaluates the log PDF element-wise for a batch `xs`.
        => logpdf, logpdf_batch, <Self::Support as Space>::Value, f64
    );
}
