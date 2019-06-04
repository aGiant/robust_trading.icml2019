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

    pub spread_mean: f64,
    pub spread_stddev: f64,

    pub rp_neutral: f64,
    pub rp_bull: f64,
    pub rp_bear: f64,

    pub drift_mean: f64,
    pub drift_stddev: f64,

    pub drift_neutral: f64,
    pub drift_bull: f64,
    pub drift_bear: f64,
}

fn mean(x: [f64; 2]) -> f64 { (x[0] - x[1]) / 2.0 }

pub fn train_value_functions<E: ExecutionDynamics>(
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

        let t = env.step(tta(quotes));

        trader.critic.handle_transition(&t.clone().replace_action(quotes));
        adversary.critic.handle_transition(&t.clone().replace_action(drift).negate_reward());

        if t.terminated() {
            break
        } else {
            drift = adversary.sample_behaviour(t.to.state());
            quotes = trader.sample_behaviour(t.to.state());
        }
    }

    env
}

pub fn train_agents_once<E: ExecutionDynamics>(
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

        let t = env.step(tta(quotes));

        trader.handle_transition(&t.clone().replace_action(quotes));
        adversary.handle_transition(&t.clone().replace_action(drift).negate_reward());

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

pub fn evaluate_agents_once<E: ExecutionDynamics>(
    mut env: Env<BrownianMotionWithDrift, E>,
    trader: &mut Trader,
    adversary: &mut Adversary
) -> (f64, f64, f64, f64, f64)
{
    let obs = env.emit();

    let mut drift = adversary.sample_target(obs.state());
    let mut quotes = trader.sample_target(obs.state());

    let mut i = 0;
    let mut drift_sum = 0.0;
    let mut reward_sum = 0.0;
    let mut spread_sum = quotes.1 * 2.0;

    loop {
        env.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0);

        let t = env.step(tta(quotes));

        drift_sum += drift;
        reward_sum += t.reward;

        if t.terminated() {
            return (env.wealth, drift_sum / i as f64, spread_sum / i as f64, reward_sum, env.inv_terminal);
        } else {
            drift = adversary.sample_target(t.to.state());
            quotes = trader.sample_target(t.to.state());

            i += 1;
            spread_sum += quotes.1 * 2.0;
        }
    }
}

pub fn evaluate_agents<E: ExecutionDynamics>(
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
    let mut average_spreads = vec![];

    for _ in 0..n_simulations {
        let (p, d, s, r, q) = evaluate_agents_once(env_builder(), trader, adversary);

        pnls.push(p);
        drifts.push(d);
        rewards.push(r);
        terminal_qs.push(q);
        average_spreads.push(s);
    }

    let pnl_est = Estimate::from_slice(&pnls);
    let dft_est = Estimate::from_slice(&drifts);
    let rwd_est = Estimate::from_slice(&rewards);
    let inv_est = Estimate::from_slice(&terminal_qs);
    let spd_est = Estimate::from_slice(&average_spreads);

    let rp_neutral = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, 0.0]))));
    let rp_bull = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, 5.0]))));
    let rp_bear = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, -5.0]))));

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

        spread_mean: spd_est.0,
        spread_stddev: spd_est.1,

        rp_neutral,
        rp_bull,
        rp_bear,

        drift_mean: dft_est.0,
        drift_stddev: dft_est.1,

        drift_neutral,
        drift_bull,
        drift_bear,
    }
}
