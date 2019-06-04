extern crate special_fun;

use crate::{
    core::{Algorithm, Parameter},
    fa::{Approximator, Embedding, Features, Parameterised, VFunction},
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use rand::{thread_rng, rngs::{ThreadRng}};
use rstat::{
    Distribution, ContinuousDistribution,
    core::Modes,
    univariate::{UnivariateMoments, continuous::Gamma as GammaDist},
};
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
use std::{fmt, ops::AddAssign, marker::PhantomData};

const MIN_TOL: f64 = 0.05;

#[derive(Clone, Debug, Serialize)]
pub struct Gamma<F> {
    alpha: F,
    beta: F,

    #[serde(skip_serializing)]
    rng: ThreadRng,
}

impl<F> Gamma<F> {
    pub fn new(alpha: F, beta: F) -> Self {
        Gamma {
            alpha, beta,

            rng: thread_rng(),
        }
    }

    #[inline]
    pub fn alpha<S>(&self, s: &S) -> f64
        where F: VFunction<S>,
    {
        self.alpha.evaluate(&self.alpha.embed(s)).unwrap() + MIN_TOL
    }

    #[inline]
    pub fn beta<S>(&self, s: &S) -> f64
        where F: VFunction<S>,
    {
        self.beta.evaluate(&self.beta.embed(s)).unwrap() + MIN_TOL
    }

    #[inline]
    fn dist<S>(&self, input: &S) -> GammaDist
        where F: VFunction<S>,
    {
        GammaDist::new(self.alpha(input), self.beta(input))
    }

    fn gl_partial(&self, alpha: f64, beta: f64, a: f64) -> [f64; 2]
        where F: Approximator<Output = f64>,
    {
        use special_fun::FloatSpecial;

        const JITTER: f64 = 1e-5;

        [beta.ln() + (a + JITTER).ln() - alpha.digamma(), alpha / beta - a]
    }
}

impl<F> Algorithm for Gamma<F> {}

impl<S, F: VFunction<S>> Policy<S> for Gamma<F> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        self.dist(input).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &S) -> f64 {
        self.dist(input).mean()
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        self.dist(input).pdf(a)
    }
}

impl<S, F: VFunction<S> + Parameterised> DifferentiablePolicy<S> for Gamma<F> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;
        let jac_alpha = self.alpha.jacobian(&phi_alpha);

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;
        let jac_beta = self.beta.jacobian(&phi_beta);

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, a);

        stack![Axis(0), gl_alpha * jac_alpha, gl_beta * jac_beta]
    }
}

impl<F: Parameterised> Parameterised for Gamma<F> {
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(0), self.alpha.weights(), self.beta.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_dim(&self) -> (usize, usize) {
        (self.alpha.weights_dim().0 + self.beta.weights_dim().0, 1)
    }
}

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Gamma<F> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, a);

        self.alpha.update(&phi_alpha, gl_alpha * error).ok();
        self.beta.update(&phi_beta, gl_beta * error).ok();
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        unimplemented!()
    }
}

impl<'de, F: Deserialize<'de>> Deserialize<'de> for Gamma<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Alpha, Beta };

        struct GammaVisitor<IF>(pub PhantomData<IF>);

        impl<'de, IF: Deserialize<'de>> Visitor<'de> for GammaVisitor<IF> {
            type Value = Gamma<IF>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Gaussain")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Gamma<IF>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let alpha = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let beta = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Gamma::new(alpha, beta))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Gamma<IF>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut alpha = None;
                let mut beta = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Alpha => {
                            if alpha.is_some() {
                                return Err(de::Error::duplicate_field("alpha"));
                            }
                            alpha = Some(map.next_value()?);
                        }
                        Field::Beta => {
                            if beta.is_some() {
                                return Err(de::Error::duplicate_field("beta"));
                            }
                            beta = Some(map.next_value()?);
                        }
                    }
                }

                let alpha = alpha.ok_or_else(|| de::Error::missing_field("alpha"))?;
                let beta = beta.ok_or_else(|| de::Error::missing_field("beta"))?;

                Ok(Gamma::new(alpha, beta))
            }
        }

        const FIELDS: &'static [&'static str] = &["alpha", "beta"];

        deserializer.deserialize_struct("Gamma", FIELDS, GammaVisitor::<F>(PhantomData))
    }
}
