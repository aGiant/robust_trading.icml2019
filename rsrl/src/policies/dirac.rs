use crate::{
    core::*,
    fa::{Approximator, VFunction, Parameterised, Embedding, Features, IntoVector},
    geometry::{MatrixView, MatrixViewMut},
    policies::{
        DifferentiablePolicy,
        ParameterisedPolicy,
        Policy
    },
};
use elementwise::arithmetic::{ElementwiseSub, ElementwiseMul};
use ndarray::Axis;
use std::ops::AddAssign;

pub struct Dirac<F> {
    pub fa: F,
}

impl<F> Dirac<F> {
    pub fn new(fa: F) -> Self {
        Dirac { fa, }
    }
}

impl<F: Parameterised> Parameterised for Dirac<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.fa.weights_view_mut()
    }
}

impl<F> Algorithm for Dirac<F> {}

impl<S, F> Policy<S> for Dirac<F>
where
    F: Approximator + Embedding<S>,
    F::Output: PartialEq,
{
    type Action = F::Output;

    fn mpa(&mut self, s: &S) -> F::Output {
        self.fa.evaluate(&self.fa.embed(s)).unwrap()
    }

    fn probability(&mut self, input: &S, a: F::Output) -> f64 {
        let mpa = self.mpa(input);

        if a.eq(&mpa) {
            1.0
        } else {
            0.0
        }
    }
}

impl<S, F> DifferentiablePolicy<S> for Dirac<F>
where
    F: Approximator + Embedding<S>,
    F::Output: ElementwiseSub + IntoVector + PartialEq,
{
    fn grad_log(&self, input: &S, a: F::Output) -> Matrix<f64> {
        let phi = self.fa.embed(input);
        let value = self.fa.evaluate(&phi).unwrap();
        let jacobian = self.fa.jacobian(&phi);

        jacobian * a.elementwise_sub(&value).into_vector().insert_axis(Axis(0))
    }
}

impl<S, F> ParameterisedPolicy<S> for Dirac<F>
where
    F: Approximator + Embedding<S> + Parameterised,
    F::Output: PartialEq + ElementwiseSub + ElementwiseMul<f64>,
{
    fn update(&mut self, input: &S, a: F::Output, error: f64) {
        unimplemented!()
        // let phi = self.fa.embed(input);
        // let value = self.fa.evaluate(&phi).unwrap();

        // self.fa.update(&phi, error).ok();
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa.weights_view_mut().add_assign(&errors);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        fa::{
            basis::fixed::Constant,
            LFA,
            VFunction,
        },
        geometry::Vector,
        policies::{Policy, ParameterisedPolicy},
    };
    use super::Dirac;

    const STATE: f64 = 0.0;

    #[test]
    fn test_f64() {
        let mut pi = Dirac::new(LFA::scalar(Constant::ones(1)));

        pi.update(&STATE, 1.0, 0.1);

        assert_eq!(pi.sample(&STATE), 0.1);
        assert_eq!(pi.probability(&STATE, 0.0), 0.0);
        assert_eq!(pi.probability(&STATE, 0.1), 1.0);
        assert_eq!(pi.probability(&STATE, 0.2), 0.0);
    }

    #[test]
    fn test_f64_pair() {
        let mut pi = Dirac::new(LFA::pair(Constant::ones(1)));

        pi.update(&STATE, [1.0, -1.0], 0.1);

        assert_eq!(pi.sample(&STATE), [0.1, -0.1]);
        assert_eq!(pi.probability(&STATE, [0.0, 0.0]), 0.0);
        assert_eq!(pi.probability(&STATE, [0.1, -0.1]), 1.0);
        assert_eq!(pi.probability(&STATE, [-0.1, 0.1]), 0.0);
    }

    #[test]
    fn test_f64_vector() {
        let mut pi = Dirac::new(LFA::vector(Constant::ones(1), 2));

        pi.update(&STATE, Vector::from_vec(vec![1.0, -1.0]), 0.1);

        assert_eq!(pi.sample(&STATE), Vector::from_vec(vec![0.1, -0.1]));
        assert_eq!(pi.probability(&STATE, Vector::from_vec(vec![0.0, 0.0])), 0.0);
        assert_eq!(pi.probability(&STATE, Vector::from_vec(vec![0.1, -0.1])), 1.0);
        assert_eq!(pi.probability(&STATE, Vector::from_vec(vec![-0.1, 0.1])), 0.0);
    }
}
