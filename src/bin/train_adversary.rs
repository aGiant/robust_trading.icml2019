extern crate algo_hft;
extern crate clap;
extern crate rand;
extern crate rsrl;
#[macro_use]
extern crate slog;
extern crate csv;

use algo_hft::{
    agents::{build_adversary, save_adversary, load_trader, training::adversary::*},
    env::Env,
};
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    logging,
};
use std::f64;

fn run_experiment(save_dir: &str, eval_interval: usize, trader_path: &str) {
    let logger = logging::root(logging::stdout());
    let mut file_logger = csv::Writer::from_path(format!("{}/results.csv", save_dir)).unwrap();

    let mut min_pnl = f64::INFINITY;
    let mut max_reward = f64::NEG_INFINITY;

    let env_builder = || Env::default_with_drift();

    // Build adversary:
    let mut trader = load_trader(trader_path.to_owned());
    let mut adversary = build_adversary(env_builder().state_space(), 0.1, 0.0001);

    // Pre-train value function:
    for _ in 0..1000 {
        train_value_function(env_builder(), &mut trader, &mut adversary);
    }

    // Run experiment:
    for i in 0..(1200*eval_interval) {
        // Perform evaluation:
        if i % eval_interval == 0 {
            let r = evaluate_adversary(
                env_builder,
                &mut trader,
                &mut adversary,
                i * eval_interval,
                1000,
            );

            // Serialise the adversary if it performed better:
            if r.wealth_mean < min_pnl || r.reward_mean > max_reward {
                min_pnl = r.wealth_mean;
                max_reward = r.reward_mean;

                save_adversary(&adversary, format!("{}/adversary_best.bin", save_dir));
            }

            // Serialise latest adversary too:
            save_adversary(&adversary, format!("{}/adversary.bin", save_dir));

            // Log plotting data:
            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => format!("{} +/- {}", r.wealth_mean, r.wealth_stddev),
                "reward" => format!("{} +/- {}", r.reward_mean, r.reward_stddev),
                "inv" => format!("{} +/- {}", r.inv_mean, r.inv_stddev),
                "drift" => format!("{} +/- {}", r.drift_mean, r.drift_stddev),
                "drift_neutral" => r.drift_neutral,
                "drift_bull" => r.drift_bull,
                "drift_bear" => r.drift_bear,
            );

            file_logger.serialize(r).ok();
            file_logger.flush().ok();
        }

        // Train adversary for one episode:
        train_adversary_once(env_builder(), &mut trader, &mut adversary);
    }
}

fn main() {
    let matches = App::new("RL adversary")
        .arg(Arg::with_name("save_dir")
                .index(1)
                .required(true))
        .arg(Arg::with_name("eval_interval")
                .index(2)
                .required(true))
        .arg(Arg::with_name("trader_path")
                .index(3)
                .required(true))
        .get_matches();

    let save_dir = matches.value_of("save_dir").unwrap();
    let eval_interval: usize = matches.value_of("eval_interval").unwrap().parse().unwrap();
    let trader_path = matches.value_of("trader_path").unwrap();

    run_experiment(save_dir, eval_interval, trader_path);
}
