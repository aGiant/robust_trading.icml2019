use crate::{
    agents::{Trader, tta},
    env::{Env, dynamics::{price::PriceDynamics, execution::ExecutionDynamics}},
    utils::Estimate,
};
use rsrl::{
    core::{Algorithm, OnlineLearner, Controller},
    domains::Domain,
    geometry::Vector,
    policies::Policy,
};

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
}

fn mean(x: [f64; 2]) -> f64 { (x[0] - x[1]) / 2.0 }

pub fn train_value_function<P: PriceDynamics, E: ExecutionDynamics>(
    mut env: Env<P, E>,
    trader: &mut Trader,
) -> Env<P, E>
{
    let mut quotes = trader.sample_behaviour(env.emit().state());

    loop {
        let t = env.step(tta(quotes));

        trader.critic.handle_transition(&t.clone().replace_action(quotes));

        if t.terminated() {
            break
        } else {
            quotes = trader.sample_behaviour(t.to.state());
        }
    }

    env
}

pub fn train_trader_once<P: PriceDynamics, E: ExecutionDynamics>(
    mut env: Env<P, E>,
    trader: &mut Trader,
) -> Env<P, E>
{
    let mut quotes = trader.sample_behaviour(env.emit().state());

    loop {
        let t = env.step(tta(quotes)).replace_action(quotes);

        trader.handle_transition(&t);

        if t.terminated() {
            break
        } else {
            quotes = trader.sample_behaviour(t.to.state());
        }
    }

    trader.handle_terminal();

    env
}

pub fn evaluate_trader_once<P: PriceDynamics, E: ExecutionDynamics>(
    mut env: Env<P, E>,
    trader: &mut Trader,
) -> (f64, f64, f64, f64)
{
    let mut quotes = trader.sample_target(env.emit().state());

    let mut i = 0;
    let mut reward_sum = 0.0;
    let mut spread_sum = quotes.1 * 2.0;

    loop {
        let t = env.step(tta(quotes));

        reward_sum += t.reward;

        if t.terminated() {
            return (env.wealth, spread_sum / i as f64, reward_sum, env.inv_terminal);
        } else {
            quotes = trader.sample_target(t.to.state());

            i += 1;
            spread_sum += quotes.1 * 2.0;
        }
    }
}

pub fn evaluate_trader<P: PriceDynamics, E: ExecutionDynamics>(
    env_builder: impl Fn() -> Env<P, E>,
    trader: &mut Trader,
    episode: usize,
    n_simulations: usize,
) -> Record
{
    let mut pnls = vec![];
    let mut rewards = vec![];
    let mut terminal_qs = vec![];
    let mut average_spread = vec![];

    for _ in 0..n_simulations {
        let (p, s, r, q) = evaluate_trader_once(env_builder(), trader);

        pnls.push(p);
        rewards.push(r);
        terminal_qs.push(q);
        average_spread.push(s);
    }

    let pnl_est = Estimate::from_slice(&pnls);
    let rwd_est = Estimate::from_slice(&rewards);
    let inv_est = Estimate::from_slice(&terminal_qs);
    let spd_est = Estimate::from_slice(&average_spread);

    let rp_neutral = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, 0.0]))));
    let rp_bull = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, 5.0]))));
    let rp_bear = mean(tta(trader.policy.mpa(&Vector::from_vec(vec![0.0, -5.0]))));

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
    }
}
