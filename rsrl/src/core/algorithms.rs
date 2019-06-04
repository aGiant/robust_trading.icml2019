#![allow(unused_variables)]
use crate::core::Shared;
use crate::domains::{Transition, Observation::Terminal};
use crate::geometry::Vector;

pub trait Algorithm {
    /// Perform housekeeping after terminal state observation.
    fn handle_terminal(&mut self) {}
}

pub trait OnlineLearner<S, A>: Algorithm {
    /// Handle a single transition collected from the problem environment.
    fn handle_transition(&mut self, transition: &Transition<S, A>);

    /// Handle an arbitrary sequence of transitions collected from the problem environment.
    fn handle_sequence(&mut self, sequence: &[Transition<S, A>]) {
        sequence.into_iter().for_each(|ref t| self.handle_transition(t));
    }
}

pub trait BatchLearner<S, A>: Algorithm {
    /// Handle a batch of samples collected from the problem environment.
    fn handle_batch(&mut self, batch: &[Transition<S, A>]);
}

pub trait Controller<S, A> {
    /// Sample the target policy for a given state `s`.
    fn sample_target(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn sample_behaviour(&mut self, s: &S) -> A;
}

pub trait ValuePredictor<S> {
    /// Compute the estimated value of V(s).
    fn predict_v(&mut self, s: &S) -> f64;
}

pub trait ActionValuePredictor<S, A>: ValuePredictor<S> {
    /// Compute the estimated value of Q(s, a).
    fn predict_qsa(&mut self, s: &S, a: A) -> f64 {
        self.predict_v(s)
    }

    /// Compute the estimated value of Q(s, ·).
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        unimplemented!()
    }
}

// Shared<T> impls:
impl<T: Algorithm> Algorithm for Shared<T> {
    fn handle_terminal(&mut self) {
        self.borrow_mut().handle_terminal()
    }
}

impl<S, A, T: OnlineLearner<S, A>> OnlineLearner<S, A> for Shared<T> {
    fn handle_transition(&mut self, transition: &Transition<S, A>) {
        self.borrow_mut().handle_transition(transition)
    }

    fn handle_sequence(&mut self, sequence: &[Transition<S, A>]) {
        self.borrow_mut().handle_sequence(sequence)
    }
}

impl<S, A, T: BatchLearner<S, A>> BatchLearner<S, A> for Shared<T> {
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        self.borrow_mut().handle_batch(batch)
    }
}

impl<S, A, T: Controller<S, A>> Controller<S, A> for Shared<T> {
    fn sample_target(&mut self, s: &S) -> A {
        self.borrow_mut().sample_target(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> A {
        self.borrow_mut().sample_behaviour(s)
    }
}

impl<S, T: ValuePredictor<S>> ValuePredictor<S> for Shared<T> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.borrow_mut().predict_v(s)
    }
}

impl<S, A, T: ActionValuePredictor<S, A>> ActionValuePredictor<S, A> for Shared<T> {
    fn predict_qsa(&mut self, s: &S, a: A) -> f64 {
        self.borrow_mut().predict_qsa(s, a)
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.borrow_mut().predict_qs(s)
    }
}
