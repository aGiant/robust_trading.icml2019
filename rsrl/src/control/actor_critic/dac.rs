// use crate::core::*;
// use crate::domains::Transition;
// use crate::fa::basis::Projector;
// use crate::geometry::Space;
// use crate::policies::{Policy, ParameterisedPolicy};
// use std::marker::PhantomData;

// /// Deterministic actor-critic.
// #[derive(Clone, Debug, Serialize, Deserialize)]
// pub struct DAC<P> {
    // pub projector: P,

    // pub weights_v: Vector<f64>,
    // pub weights_w: Vector<f64>,
    // pub weights_theta: Matrix<f64>,

    // pub alpha: Parameter,
    // pub beta: Parameter,
    // pub gamma: Parameter,
// }

// impl<P: Space> DAC<P> {
    // pub fn new<T1, T2, T3>(projector: P, action_dim: usize, alpha: T1, beta: T2, gamma: T3) -> Self
    // where
        // T1: Into<Parameter>,
        // T2: Into<Parameter>,
        // T3: Into<Parameter>,
    // {
        // let dim = projector.dim();

        // DAC {
            // projector,

            // weights_v: Vector::zeros(dim),
            // weights_w: Vector::zeros(dim),
            // weights_theta: Matrix::zeros((dim, action_dim)),

            // alpha: alpha.into(),
            // beta: beta.into(),
            // gamma: gamma.into(),
        // }
    // }
// }

// impl<P> Algorithm for DAC<P> {
    // fn handle_terminal(&mut self) {
        // self.alpha = self.alpha.step();
        // self.gamma = self.gamma.step();
    // }
// }

// impl<S, P: Projector<S>> OnlineLearner<S, Vector<f64>> for DAC<P> {
    // fn handle_transition(&mut self, t: &Transition<S, Vector<f64>>) {
        // let s = t.from.state();
        // let phi = self.projector.project(s);

        // let v = self.critic.predict_v(s);
        // let td_error = if t.terminated() {
            // t.reward - v
        // } else {
            // t.reward + self.gamma * self.predict_v(t.to.state()) - v
        // };

        // self.critic.handle_transition(t);
        // self.policy.update(s, t.action.clone(), self.alpha * td_error);
    // }
// }

// impl<S, P> ValuePredictor<S> for DAC<P> {
    // fn predict_v(&mut self, s: &S) -> f64 {
        // self.critic.predict_v(s)
    // }
// }

// impl<S, P> ActionValuePredictor<S, Vector<f64>> for DAC<P> {
    // fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        // self.critic.predict_qs(s)
    // }

    // fn predict_qsa(&mut self, s: &S, a: Vector<f64>) -> f64 {
        // self.critic.predict_qsa(s, a)
    // }
// }

// impl<S, P> Controller<S, Vector<f64>> for DAC<P> {
    // fn sample_target(&mut self, s: &S) -> Vector<f64> {
        // self.policy.sample(s)
    // }

    // fn sample_behaviour(&mut self, s: &S) -> Vector<f64> {
        // self.policy.sample(s)
    // }
// }
