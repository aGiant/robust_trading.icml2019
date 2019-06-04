use crate::core::*;
use crate::domains::Transition;
use crate::fa::*;
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Greedy, Policy, FinitePolicy};

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
pub struct GreedyGQ<Q, W, PB> {
    pub fa_q: Q,
    pub fa_w: W,

    pub target_policy: Greedy<Q>,
    pub behaviour_policy: PB,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<Q, W, PB> GreedyGQ<Shared<Q>, W, PB> {
    pub fn new<P1, P2, P3>(
        fa_q: Q,
        fa_w: W,
        behaviour_policy: PB,
        alpha: P1,
        beta: P2,
        gamma: P3,
    ) -> Self
    where
        P1: Into<Parameter>,
        P2: Into<Parameter>,
        P3: Into<Parameter>,
    {
        let fa_q = make_shared(fa_q);

        GreedyGQ {
            fa_q: fa_q.clone(),
            fa_w,

            target_policy: Greedy::new(fa_q),
            behaviour_policy,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, W, PB> Algorithm for GreedyGQ<Q, W, PB> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, Q, W, PB> OnlineLearner<S, PB::Action> for GreedyGQ<Q, W, PB>
where
    Q: QFunction<S>,
    W: VFunction<S>,
    PB: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, PB::Action>) {
        let s = t.from.state();
        let phi_s = self.fa_w.embed(s);
        let estimate = self.fa_w.evaluate(&phi_s).unwrap();

        if t.terminated() {
            let residual = t.reward - self.fa_q.evaluate_index(&phi_s, t.action).unwrap();

            self.fa_w.update(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            ).ok();
            self.fa_q.update_index(
                &phi_s,
                t.action,
                self.alpha.value() * residual
            ).ok();
        } else {
            let ns = t.to.state();
            let na = self.sample_target(ns);
            let phi_ns = self.fa_w.embed(ns);

            let residual =
                t.reward
                + self.gamma.value() * self.fa_q.evaluate_index(&phi_ns, na).unwrap()
                - self.fa_q.evaluate_index(&phi_s, t.action).unwrap();

            let n_features = self.fa_q.n_features();
            let update_q = residual * phi_s.clone().expanded(n_features)
                - estimate * self.gamma.value() * phi_ns.expanded(n_features);

            self.fa_w.update(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            ).ok();
            self.fa_q.update_index(
                &Features::Dense(update_q),
                t.action,
                self.alpha.value()
            ).ok();
        }
    }
}

impl<S, Q, W, PB> ValuePredictor<S> for GreedyGQ<Q, W, PB>
where
    Q: QFunction<S>,
    PB: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.target_policy.probabilities(s))
    }
}

impl<S, Q, W, PB> ActionValuePredictor<S, PB::Action> for GreedyGQ<Q, W, PB>
where
    Q: QFunction<S>,
    PB: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_q.evaluate(&self.fa_q.embed(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.fa_q.evaluate_index(&self.fa_q.embed(s), a).unwrap()
    }
}

impl<S, Q, W, PB> Controller<S, PB::Action> for GreedyGQ<Q, W, PB>
where
    Q: QFunction<S>,
    PB: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> PB::Action {
        self.target_policy.sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> PB::Action {
        self.behaviour_policy.sample(s)
    }
}

impl<Q: Parameterised, W, PB> Parameterised for GreedyGQ<Q, W, PB> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_q.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_q.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.fa_q.weights_view_mut()
    }
}
