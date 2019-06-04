use crate::{
    env::dynamics::{
        ASDynamics,
        execution::{ExecutionDynamics, PoissonRate},
        price::{PriceDynamics, BrownianMotion, BrownianMotionWithDrift},
    },
};
use rand::thread_rng;
use rsrl::{
    domains::{Domain, Transition, Observation},
    geometry::{
        continuous::{Interval, Reals},
        product::{DoubleSpace, LinearSpace},
        Vector,
    },
};

pub mod dynamics;
pub mod strategies;

const INV_BOUNDS: [f64; 2] = [-50.0, 50.0];

#[derive(Debug)]
pub struct Env<P, E> {
    pub dynamics: ASDynamics<P, E>,

    pub inv: f64,
    pub inv_terminal: f64,

    pub ask_executed: bool,
    pub bid_executed: bool,

    pub reward: f64,
    pub wealth: f64,
}

impl Env<BrownianMotion, PoissonRate> {
    pub fn default() -> Self {
        Self::new(ASDynamics::new(
            0.005, 100.0, thread_rng(),
            BrownianMotion::new(0.005, 2.0),
            PoissonRate::default()
        ))
    }
}

impl Env<BrownianMotionWithDrift, PoissonRate> {
    pub fn default_with_drift() -> Self {
        Self::new(ASDynamics::new(
            0.005, 100.0, thread_rng(),
            BrownianMotionWithDrift::new(0.005, 0.0, 2.0),
            PoissonRate::default()
        ))
    }
}

impl<P: PriceDynamics, E: ExecutionDynamics> Env<P, E> {
    pub fn new(dynamics: ASDynamics<P, E>) -> Self {
        Self {
            dynamics,

            inv: 0.0,
            inv_terminal: 0.0,

            ask_executed: false,
            bid_executed: false,

            reward: 0.0,
            wealth: 0.0,
        }
    }

    fn do_executions(&mut self, ask_price: f64, bid_price: f64) {
        if self.inv > INV_BOUNDS[0] {
            if let Some(ask_offset) = self.dynamics.try_execute_ask(ask_price) {
                self.ask_executed = true;
                self.inv -= 1.0;
                self.reward += ask_offset;
                self.wealth += ask_price;
            }
        }

        if self.inv < INV_BOUNDS[1] {
            if let Some(bid_offset) = self.dynamics.try_execute_bid(bid_price) {
                self.bid_executed = true;
                self.inv += 1.0;
                self.reward += bid_offset;
                self.wealth -= bid_price;
            }
        }
    }

    fn update_state(&mut self, ask_offset: f64, bid_offset: f64) {
        let ask_price = self.dynamics.price + ask_offset;
        let bid_price = self.dynamics.price - bid_offset;

        self.reward = self.inv * self.dynamics.innovate();
        self.ask_executed = false;
        self.bid_executed = false;

        self.do_executions(ask_price, bid_price);

        if self.is_terminal() {
            // Execute market order favourably at midprice:
            self.wealth += self.dynamics.price * self.inv;
            self.reward -= 0.5 * self.inv.powi(2);

            self.inv_terminal = self.inv;
            self.inv = 0.0;
        }
    }
}

impl<P: PriceDynamics, E: ExecutionDynamics> Domain for Env<P, E> {
    type StateSpace = LinearSpace<Interval>;
    type ActionSpace = DoubleSpace<Reals>;

    fn emit(&self) -> Observation<Vector<f64>> {
        let state = vec![self.dynamics.time, self.inv.min(INV_BOUNDS[1]).max(INV_BOUNDS[0])];

        if self.is_terminal() {
            Observation::Terminal(state.into())
        } else {
            Observation::Full(state.into())
        }
    }

    fn step(&mut self, action: [f64; 2]) -> Transition<Vector<f64>, [f64; 2]> {
        let from = self.emit();

        self.update_state(action[0], action[1]);

        let to = self.emit();
        let reward = self.reward(&from, &to);

        Transition {
            from,
            action,
            reward,
            to,
        }
    }

    fn is_terminal(&self) -> bool {
        self.dynamics.time >= 1.0
    }

    fn reward(&self, _: &Observation<Vector<f64>>, _: &Observation<Vector<f64>>) -> f64 {
        self.reward
    }

    fn state_space(&self) -> Self::StateSpace {
        LinearSpace::empty()
            + Interval::bounded(0.0, 1.0)
            + Interval::bounded(INV_BOUNDS[0], INV_BOUNDS[1])
    }

    fn action_space(&self) -> Self::ActionSpace {
        DoubleSpace::new(Reals, Reals)
    }
}
