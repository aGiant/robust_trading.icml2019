use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, Approximator, VFunction};
use crate::geometry::{Matrix, MatrixView, MatrixViewMut};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarianceTD<J, V> {
    pub value_estimator: J,
    pub variance_estimator: V,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<J, V> VarianceTD<J, V> {
    pub fn new<T1, T2>(value_estimator: J, variance_estimator: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        VarianceTD {
            value_estimator,
            variance_estimator,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<J, V> VarianceTD<J, V> {
    fn compute_value_error<S>(&self, s: &S, reward: f64, ns: &S) -> f64
    where
        J: VFunction<S>,
    {
        reward + self.value_estimator.state_value(ns) - self.value_estimator.state_value(s)
    }
}

impl<J, V> Algorithm for VarianceTD<J, V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, J: VFunction<S>, V: VFunction<S>> OnlineLearner<S, A> for VarianceTD<J, V> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let value_error = self.compute_value_error(t.from.state(), t.reward, t.to.state());
        let meta_reward = value_error * value_error;

        let phi_s = self.variance_estimator.embed(t.from.state());
        let v = self.variance_estimator.evaluate(&phi_s).unwrap();

        let td_error = if t.terminated() {
            meta_reward - v
        } else {
            let gamma_var = self.gamma * self.gamma;

            meta_reward + gamma_var * self.predict_v(t.to.state()) - v
        };

        self.variance_estimator.update(&phi_s, self.alpha * td_error).ok();
    }
}

impl<S, J, V: VFunction<S>> ValuePredictor<S> for VarianceTD<J, V> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.variance_estimator.state_value(s)
    }
}

impl<S, A, J: VFunction<S>, V: VFunction<S>> ActionValuePredictor<S, A> for VarianceTD<J, V> {}

impl<J, V: Parameterised> Parameterised for VarianceTD<J, V> {
    fn weights(&self) -> Matrix<f64> {
        self.variance_estimator.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.variance_estimator.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.variance_estimator.weights_view_mut()
    }
}
