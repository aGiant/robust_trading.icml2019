extern crate algo_hft;
extern crate clap;
extern crate rsrl;
#[macro_use]
extern crate slog;
extern crate csv;

use algo_hft::{
    agents::{build_trader, save_trader, training::trader::*},
    env::Env,
};
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    logging,
};
use std::f64;

fn run_experiment(save_dir: &str, eval_interval: usize, _risk_param: Option<f64>) {
    let logger = logging::root(logging::stdout());
    let mut file_logger = csv::Writer::from_path(format!("{}/results.csv", save_dir)).unwrap();

    let mut max_pnl = f64::NEG_INFINITY;
    let mut max_reward = f64::NEG_INFINITY;

    let env_builder = || Env::default();

    // Build trader:
    let mut trader = build_trader(env_builder().state_space(), 0.01, 0.000001);

    // Pre-train value function:
    for _ in 0..1000 {
        train_value_function(env_builder(), &mut trader);
    }

    // Run experiment:
    for i in 0..(1200*eval_interval) {
        // Perform evaluation:
        if i % eval_interval == 0 {
            let r = evaluate_trader(
                env_builder,
                &mut trader,
                i * eval_interval,
                1000,
            );

            // Serialise the trader if it performed better:
            if r.wealth_mean > max_pnl || r.reward_mean > max_reward {
                max_pnl = r.wealth_mean;
                max_reward = r.reward_mean;

                save_trader(&trader, format!("{}/trader_best.bin", save_dir));
            }

            // Serialise latest trader too:
            save_trader(&trader, format!("{}/trader.bin", save_dir));

            // Log plotting data:
            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => format!("{} +/- {}", r.wealth_mean, r.wealth_stddev),
                "reward" => format!("{} +/- {}", r.reward_mean, r.reward_stddev),
                "inv" => format!("{} +/- {}", r.inv_mean, r.inv_stddev),
                "spread" => format!("{} +/- {}", r.spread_mean, r.spread_stddev),
                "rp_neutral" => r.rp_neutral,
                "rp_bull" => r.rp_bull,
                "rp_bear" => r.rp_bear,
            );

            file_logger.serialize(r).ok();
            file_logger.flush().ok();
        }

        // Train trader for one episode:
        train_trader_once(env_builder(), &mut trader);
    }
}

fn main() {
    let matches = App::new("RL trader")
        .arg(Arg::with_name("save_dir")
                .index(1)
                .required(true))
        .arg(Arg::with_name("eval_interval")
                .index(2)
                .required(true))
        .arg(Arg::with_name("risk_param")
                .long("risk_param")
                .required(false))
        .get_matches();

    let save_dir = matches.value_of("save_dir").unwrap();
    let eval_interval: usize = matches.value_of("eval_interval").unwrap().parse().unwrap();
    let risk_param: Option<f64> = matches.value_of("risk_param").map(|s| s.parse().unwrap());

    run_experiment(save_dir, eval_interval, risk_param);
}
