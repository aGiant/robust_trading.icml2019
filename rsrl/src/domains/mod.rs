//! Learning benchmark domains module.
use crate::geometry::Space;

macro_rules! impl_into {
    (Transition < S, $type:ty > => Transition < S,() >) => {
        impl<S> Into<Transition<S, ()>> for Transition<S, $type> {
            fn into(self) -> Transition<S, ()> { self.drop_action() }
        }
    };
}

/// Container class for data associated with a domain observation.
#[derive(Clone, Copy, Debug)]
pub enum Observation<S> {
    /// Fully observed state of the environment.
    Full(S),

    /// Partially observed state of the environment.
    Partial(S),

    /// Terminal state of the environment.
    Terminal(S),
}

impl<S> Observation<S> {
    /// Helper function returning a reference to the state values for the given
    /// observation.
    pub fn state(&self) -> &S {
        use self::Observation::*;

        match self {
            Full(ref state) | Partial(ref state) | Terminal(ref state) => state,
        }
    }

    /// Returns true if the state was fully observed, otherwise false.
    pub fn is_full(&self) -> bool {
        match self {
            Observation::Full(_) => true,
            _ => false,
        }
    }

    /// Returns true if the state was only partially observed, otherwise false.
    pub fn is_partial(&self) -> bool {
        match self {
            Observation::Partial(_) => true,
            _ => false,
        }
    }

    /// Returns true if the observation is the terminal state, otherwise false.
    pub fn is_terminal(&self) -> bool {
        match self {
            Observation::Terminal(_) => true,
            _ => false,
        }
    }
}

/// Container class for data associated with a domain transition.
#[derive(Clone, Copy, Debug)]
pub struct Transition<S, A> {
    /// State transitioned _from_, `s`.
    pub from: Observation<S>,

    /// Action taken to initiate the transition (control tasks).
    pub action: A,

    /// Reward obtained from the transition.
    pub reward: f64,

    /// State transitioned _to_, `s'`.
    pub to: Observation<S>,
}

impl<S, A> Transition<S, A> {
    /// Return references to the `from` and `to` states associated with this
    /// transition.
    pub fn states(&self) -> (&S, &S) { (self.from.state(), self.to.state()) }

    /// Apply a closure to the `from` and `to` states associated with this
    /// transition.
    pub fn map_states<O>(&self, f: impl Fn(&S) -> O) -> (O, O) {
        (f(self.from.state()), f(self.to.state()))
    }

    /// Returns true if the transition ends in a terminal state.
    pub fn terminated(&self) -> bool { self.to.is_terminal() }

    /// Replace the action associated with this transition and return a new
    /// instance.
    pub fn replace_action<T>(self, action: T) -> Transition<S, T> {
        Transition {
            from: self.from,
            action: action,
            reward: self.reward,
            to: self.to,
        }
    }

    /// Drop the action associated with this transition and return a new
    /// instance.
    pub fn drop_action(self) -> Transition<S, ()> { self.replace_action(()) }

    pub fn replace_reward(self, r: f64) -> Transition<S, A> {
        Transition {
            from: self.from,
            action: self.action,
            reward: r,
            to: self.to,
        }
    }

    pub fn negate_reward(self) -> Transition<S, A> {
        Transition {
            from: self.from,
            action: self.action,
            reward: -self.reward,
            to: self.to,
        }
    }
}

impl_into!(Transition<S, u8> => Transition<S, ()>);
impl_into!(Transition<S, u16> => Transition<S, ()>);
impl_into!(Transition<S, u32> => Transition<S, ()>);
impl_into!(Transition<S, u64> => Transition<S, ()>);
impl_into!(Transition<S, usize> => Transition<S, ()>);
impl_into!(Transition<S, i8> => Transition<S, ()>);
impl_into!(Transition<S, i16> => Transition<S, ()>);
impl_into!(Transition<S, i32> => Transition<S, ()>);
impl_into!(Transition<S, i64> => Transition<S, ()>);
impl_into!(Transition<S, isize> => Transition<S, ()>);
impl_into!(Transition<S, f32> => Transition<S, ()>);
impl_into!(Transition<S, f64> => Transition<S, ()>);

/// An interface for constructing reinforcement learning problem domains.
pub trait Domain {
    /// State space representation type class.
    type StateSpace: Space;

    /// Action space representation type class.
    type ActionSpace: Space;

    /// Emit an observation of the current state of the environment.
    fn emit(&self) -> Observation<<Self::StateSpace as Space>::Value>;

    /// Transition the environment forward a single step given an action, `a`.
    fn step(
        &mut self,
        a: <Self::ActionSpace as Space>::Value,
    ) -> Transition<<Self::StateSpace as Space>::Value, <Self::ActionSpace as Space>::Value>;

    /// Returns true if the current state is terminal.
    fn is_terminal(&self) -> bool;

    /// Compute the reward associated with a transition from one state to
    /// another.
    fn reward(
        &self,
        from: &Observation<<Self::StateSpace as Space>::Value>,
        to: &Observation<<Self::StateSpace as Space>::Value>,
    ) -> f64;

    /// Returns an instance of the state space type class.
    fn state_space(&self) -> Self::StateSpace;

    /// Returns an instance of the action space type class.
    fn action_space(&self) -> Self::ActionSpace;
}

mod ode;
use self::ode::*;

mod grid_world;

import_all!(mountain_car);
import_all!(cart_pole);
import_all!(acrobat);
import_all!(hiv);
import_all!(cliff_walk);

#[cfg(feature = "openai")]
import_all!(openai);
