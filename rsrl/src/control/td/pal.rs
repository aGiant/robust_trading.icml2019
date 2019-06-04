use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Greedy, Policy};

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<Q, P> {
    pub q_func: Q,

    pub policy: P,
    pub target: Greedy<Q>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> PAL<Shared<Q>, P> {
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        let q_func = make_shared(q_func);

        PAL {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for PAL<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for PAL<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.q_func.embed(s);
        let qs = self.q_func.evaluate(&phi_s).unwrap();

        let residual = if t.terminated() {
            t.reward - qs[t.action]
        } else {
            let ns = t.to.state();
            let nqs = self.predict_qs(ns);

            let a_star = self.sample_target(s);
            let na_star = self.sample_target(ns);

            let td_error = t.reward + self.gamma * nqs[a_star] - qs[t.action];
            let al_error = td_error - self.alpha * (qs[a_star] - qs[t.action]);

            al_error.max(td_error - self.alpha * (nqs[na_star] - nqs[t.action]))
        };

        self.q_func.update_index(&phi_s, t.action, self.alpha * residual).ok();
    }
}

impl<S, Q, P> Controller<S, P::Action> for PAL<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.sample(s) }
}

impl<S, Q, P> ValuePredictor<S> for PAL<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for PAL<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.evaluate(&self.q_func.embed(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate_index(&self.q_func.embed(s), a).unwrap()
    }
}

impl<Q: Parameterised, P> Parameterised for PAL<Q, P> {
    fn weights(&self) -> Matrix<f64> {
        self.q_func.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.q_func.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.q_func.weights_view_mut()
    }
}
