use crate::core::*;
use crate::domains::Transition;
use crate::geometry::Matrix;
use crate::fa::Parameterised;
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

pub struct BaselineREINFORCE<B, P> {
    pub policy: P,
    pub baseline: B,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<B, P> BaselineREINFORCE<B, P> {
    pub fn new<T1, T2>(policy: P, baseline: B, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        BaselineREINFORCE {
            policy,
            baseline,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<B, P> Algorithm for BaselineREINFORCE<B, P>
where
    B: Algorithm,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
        self.baseline.handle_terminal();
    }
}

impl<S, B, P> BatchLearner<S, P::Action> for BaselineREINFORCE<B, P>
where
    S: Clone,
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
    B: BatchLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
{
    fn handle_batch(&mut self, batch: &[Transition<S, P::Action>]) {
        self.baseline.handle_batch(batch);

        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            let s = t.from.state();
            let baseline = self.baseline.predict_qsa(s, t.action.clone());

            ret = t.reward + self.gamma * ret;

            self.policy.update(s, t.action.clone(), self.alpha * (ret - baseline));
        }
    }
}

impl<S, B, P: ParameterisedPolicy<S>> Controller<S, P::Action> for BaselineREINFORCE<B, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.sample(s) }
}

impl<B, P: Parameterised> Parameterised for BaselineREINFORCE<B, P> {
    fn weights(&self) -> Matrix<f64> {
        self.policy.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.policy.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.policy.weights_view_mut()
    }
}
