use crate::domains::{Domain, Observation, Transition};
use crate::geometry::{
    Surjection,
    Vector,
    continuous::Interval,
    product::LinearSpace,
};

const X_MIN: f64 = -1.2;
const X_MAX: f64 = 0.6;

const V_MIN: f64 = -0.07;
const V_MAX: f64 = 0.07;

const FORCE_G: f64 = -0.0025;
const FORCE_CAR: f64 = 0.0015;

const HILL_FREQ: f64 = 3.0;

const REWARD_STEP: f64 = -1.0;
const REWARD_GOAL: f64 = 0.0;

const MIN_ACTION: f64 = -1.0;
const MAX_ACTION: f64 = 1.0;

pub struct ContinuousMountainCar {
    x: f64,
    v: f64,

    action_space: Interval,
}

impl ContinuousMountainCar {
    fn new(x: f64, v: f64) -> ContinuousMountainCar {
        ContinuousMountainCar {
            x, v,
            action_space: Interval::bounded(MIN_ACTION, MAX_ACTION),
        }
    }

    fn dv(x: f64, a: f64) -> f64 { FORCE_CAR * a + FORCE_G * (HILL_FREQ * x).cos() }

    fn update_state(&mut self, a: f64) {
        let a = self.action_space.map(a);

        self.v = clip!(V_MIN, self.v + Self::dv(self.x, a), V_MAX);
        self.x = clip!(X_MIN, self.x + self.v, X_MAX);
    }
}

impl Default for ContinuousMountainCar {
    fn default() -> ContinuousMountainCar { ContinuousMountainCar::new(-0.5, 0.0) }
}

impl Domain for ContinuousMountainCar {
    type StateSpace = LinearSpace<Interval>;
    type ActionSpace = Interval;

    fn emit(&self) -> Observation<Vector<f64>> {
        let s = Vector::from_vec(vec![self.x, self.v]);

        if self.is_terminal() {
            Observation::Terminal(s)
        } else {
            Observation::Full(s)
        }
    }

    fn step(&mut self, action: f64) -> Transition<Vector<f64>, f64> {
        let from = self.emit();

        self.update_state(action);
        let to = self.emit();
        let reward = self.reward(&from, &to);

        Transition {
            from,
            action,
            reward,
            to,
        }
    }

    fn is_terminal(&self) -> bool { self.x >= X_MAX }

    fn reward(&self, _: &Observation<Vector<f64>>, to: &Observation<Vector<f64>>) -> f64 {
        match *to {
            Observation::Terminal(_) => REWARD_GOAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        LinearSpace::empty() + Interval::bounded(X_MIN, X_MAX) + Interval::bounded(V_MIN, V_MAX)
    }

    fn action_space(&self) -> Interval { Interval::bounded(MIN_ACTION, MAX_ACTION) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = ContinuousMountainCar::default();

        match m.emit() {
            Observation::Full(ref state) => {
                assert_eq!(state[0], -0.5);
                assert_eq!(state[1], 0.0);
            },
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_is_terminal() {
        assert!(!ContinuousMountainCar::default().is_terminal());
        assert!(!ContinuousMountainCar::new(-0.5, 0.0).is_terminal());

        assert!(ContinuousMountainCar::new(X_MAX, -0.05).is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX, 0.0).is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX, 0.05).is_terminal());

        assert!(!ContinuousMountainCar::new(X_MAX - 0.0001 * X_MAX, 0.0).is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX + 0.0001 * X_MAX, 0.0).is_terminal());
    }

    #[test]
    fn test_reward() {
        let mc = ContinuousMountainCar::default();

        let s = mc.emit();
        let ns = ContinuousMountainCar::new(X_MAX, 0.0).emit();

        assert_eq!(mc.reward(&s, &s), REWARD_STEP);
        assert_eq!(mc.reward(&s, &ns), REWARD_GOAL);
    }
}
