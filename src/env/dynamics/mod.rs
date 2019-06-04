use rand::{ThreadRng, thread_rng, prelude::*};

pub mod price;
pub mod execution;

#[derive(Debug)]
pub struct ASDynamics<P, E> {
    rng: ThreadRng,

    pub dt: f64,
    pub time: f64,
    pub price: f64,
    pub price_initial: f64,

    pub price_dynamics: P,
    pub execution_dynamics: E,
}

impl<P, E> ASDynamics<P, E> {
    pub fn new(dt: f64, price: f64, rng: ThreadRng,
               price_dynamics: P, execution_dynamics: E) -> Self
    {
        ASDynamics {
            rng,

            dt,
            time: 0.0,
            price,
            price_initial: price,

            price_dynamics,
            execution_dynamics,
        }
    }
}

impl ASDynamics<price::BrownianMotionWithDrift, execution::PoissonRate> {
    pub fn default_with_drift(drift: f64) -> Self {
        const DT: f64 = 0.005;

        let pd = price::BrownianMotionWithDrift::new(DT, drift, 2.0);
        let ed = execution::PoissonRate::new(DT, 140.0, 1.5);

        ASDynamics::new(DT, 100.0, thread_rng(), pd, ed)
    }
}

impl Default for ASDynamics<price::BrownianMotion, execution::PoissonRate> {
    fn default() -> Self {
        const DT: f64 = 0.005;

        let pd = price::BrownianMotion::new(DT, 2.0);
        let ed = execution::PoissonRate::new(DT, 140.0, 1.5);

        ASDynamics::new(DT, 100.0, thread_rng(), pd, ed)
    }
}

impl<P, E> ASDynamics<P, E>
where
    P: price::PriceDynamics,
    E: execution::ExecutionDynamics,
{
    pub fn innovate(&mut self) -> f64 {
        let mut rng = thread_rng();

        let price_inc = self.price_dynamics.sample_increment(&mut rng, self.price);

        self.time += self.dt;
        self.price += price_inc;

        price_inc
    }

    fn try_execute(&mut self, offset: f64) -> Option<f64> {
        let match_prob = self.execution_dynamics.match_prob(offset);

        if self.rng.gen_bool(match_prob) {
            Some(offset)
        } else {
            None
        }
    }

    pub fn try_execute_ask(&mut self, order_price: f64) -> Option<f64> {
        let offset = order_price - self.price;

        self.try_execute(offset)
    }

    pub fn try_execute_bid(&mut self, order_price: f64) -> Option<f64> {
        let offset = self.price - order_price;

        self.try_execute(offset)
    }
}
