use crate::{
    agents::{Trader, Adversary, tta},
    env::{Env, dynamics::{price::BrownianMotionWithDrift, execution::ExecutionDynamics}},
    utils::Estimate,
};
use rsrl::{
    core::{Algorithm, OnlineLearner, Controller},
    domains::Domain,
    geometry::Vector,
    policies::Policy,
};

const MAX_DRIFT: f64 = 5.0;

#[derive(Debug, Serialize)]
pub struct Record {
    pub episode: usize,

    pub wealth_mean: f64,
    pub wealth_stddev: f64,

    pub reward_mean: f64,
    pub reward_stddev: f64,

    pub inv_mean: f64,
    pub inv_stddev: f64,

    pub drift_mean: f64,
    pub drift_stddev: f64,

    pub drift_neutral: f64,
    pub drift_bull: f64,
    pub drift_bear: f64,
}

pub fn train_value_function<E: ExecutionDynamics>(
    mut env: Env<BrownianMotionWithDrift, E>,
    trader: &mut Trader,
    adversary: &mut Adversary
) -> Env<BrownianMotionWithDrift, E>
{
    let obs = env.emit();

    let mut drift = adversary.sample_behaviour(obs.state());
    let mut quotes = trader.sample_behaviour(obs.state());

    loop {
        env.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0);

        let t = env.step(tta(quotes)).replace_action(drift).negate_reward();

        adversary.critic.handle_transition(&t);

        if t.terminated() {
            break
        } else {
            drift = adversary.sample_behaviour(t.to.state());
            quotes = trader.sample_behaviour(t.to.state());
        }
    }

    env
}

pub fn train_adversary_once<E: ExecutionDynamics>(
    mut env: Env<BrownianMotionWithDrift, E>,
    trader: &mut Trader,
    adversary: &mut Adversary
) -> Env<BrownianMotionWithDrift, E>
{
    let obs = env.emit();

    let mut drift = adversary.sample_behaviour(obs.state());
    let mut quotes = trader.sample_behaviour(obs.state());

    loop {
        env.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0);

        let t = env.step(tta(quotes)).replace_action(drift).negate_reward();

        adversary.handle_transition(&t);

        if t.terminated() {
            break
        } else {
            drift = adversary.sample_behaviour(t.to.state());
            quotes = trader.sample_behaviour(t.to.state());
        }
    }

    trader.handle_terminal();
    adversary.handle_terminal();

    env
}

pub fn evaluate_adversary_once<E: ExecutionDynamics>(
    mut env: Env<BrownianMotionWithDrift, E>,
    trader: &mut Trader,
    adversary: &mut Adversary
) -> (f64, f64, f64, f64)
{
    let mut i = 0;
    let mut drift_sum = 0.0;
    let mut reward_sum = 0.0;

    let obs = env.emit();

    let mut drift = adversary.sample_target(obs.state());
    let mut quotes = trader.sample_target(obs.state());

    loop {
        env.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0);

        let t = env.step(tta(quotes));

        i += 1;
        drift_sum += drift;
        reward_sum += t.reward;

        if t.terminated() {
            return (env.wealth, drift_sum / i as f64, reward_sum, env.inv_terminal);
        } else {
            drift = adversary.sample_target(t.to.state());
            quotes = trader.sample_target(t.to.state());
        }
    }
}

pub fn evaluate_adversary<E: ExecutionDynamics>(
    env_builder: impl Fn() -> Env<BrownianMotionWithDrift, E>,
    trader: &mut Trader,
    adversary: &mut Adversary,
    episode: usize,
    n_simulations: usize,
) -> Record
{
    let mut pnls = vec![];
    let mut drifts = vec![];
    let mut rewards = vec![];
    let mut terminal_qs = vec![];

    for _ in 0..n_simulations {
        let (p, d, r, q) = evaluate_adversary_once(env_builder(), trader, adversary);

        pnls.push(p);
        drifts.push(d);
        rewards.push(r);
        terminal_qs.push(q);
    }

    let pnl_est = Estimate::from_slice(&pnls);
    let rwd_est = Estimate::from_slice(&rewards);
    let inv_est = Estimate::from_slice(&terminal_qs);
    let dft_est = Estimate::from_slice(&drifts);

    let drift_neutral = adversary.policy.mpa(&Vector::from_vec(vec![0.0, 0.0]));
    let drift_bull = adversary.policy.mpa(&Vector::from_vec(vec![0.0, 5.0]));
    let drift_bear = adversary.policy.mpa(&Vector::from_vec(vec![0.0, -5.0]));

    Record {
        episode,

        wealth_mean: pnl_est.0,
        wealth_stddev: pnl_est.1,

        reward_mean: rwd_est.0,
        reward_stddev: rwd_est.1,

        inv_mean: inv_est.0,
        inv_stddev: inv_est.1,

        drift_mean: dft_est.0,
        drift_stddev: dft_est.1,

        drift_neutral,
        drift_bull,
        drift_bear,
    }
}
