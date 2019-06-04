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
    univariate::{UnivariateMoments, continuous::Beta as BetaDist},
};
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
use std::{fmt::{self, Debug}, marker::PhantomData, ops::AddAssign};

const MIN_TOL: f64 = 1.0;

#[derive(Clone, Debug, Serialize)]
pub struct Beta<F> {
    alpha: F,
    beta: F,

    #[serde(skip_serializing)]
    rng: ThreadRng,
}

impl<F> Beta<F> {
    pub fn new(alpha: F, beta: F) -> Self {
        Beta {
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
    fn dist<S>(&self, input: &S) -> BetaDist
        where F: VFunction<S>,
    {
        BetaDist::new(self.alpha(input), self.beta(input))
    }

    fn gl_partial(&self, alpha: f64, beta: f64, a: f64) -> [f64; 2]
        where F: Approximator<Output = f64>,
    {
        use special_fun::FloatSpecial;

        const JITTER: f64 = 1e-5;

        let apb_digamma = (alpha + beta).digamma();
        let alpha_digamma = alpha.digamma();
        let beta_digamma = beta.digamma();

        [
            (a + JITTER).ln() - alpha_digamma + apb_digamma,
            (1.0 - a + JITTER).ln() - beta_digamma + apb_digamma
        ]
    }
}

impl<F> Algorithm for Beta<F> {}

impl<S, F: VFunction<S>> Policy<S> for Beta<F> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        self.dist(input).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &S) -> f64 {
        let d = self.dist(input);
        let modes = d.modes();

        if modes.len() == 0 { d.mean() } else { modes[0] }
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        self.dist(input).pdf(a)
    }
}

impl<S, F: VFunction<S> + Parameterised> DifferentiablePolicy<S> for Beta<F> {
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

impl<F: Parameterised> Parameterised for Beta<F> {
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

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Beta<F> {
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

impl<'de, F> Deserialize<'de> for Beta<F>
where
    F: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Alpha, Beta };

        struct BetaVisitor<IF>(pub PhantomData<IF>);

        impl<'de, IF> Visitor<'de> for BetaVisitor<IF>
        where
            IF: Deserialize<'de>,
        {
            type Value = Beta<IF>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Beta")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Beta<IF>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let alpha = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let beta = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Beta::new(alpha, beta))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Beta<IF>, V::Error>
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

                Ok(Beta::new(alpha, beta))
            }
        }

        const FIELDS: &'static [&'static str] = &["alpha", "beta"];

        deserializer.deserialize_struct(
            "Beta",
            FIELDS,
            BetaVisitor::<F>(PhantomData)
        )
    }
}
