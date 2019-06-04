extern crate algo_hft;
extern crate clap;
extern crate rsrl;
#[macro_use]
extern crate slog;
extern crate csv;

use algo_hft::{
    agents::{
        build_adversary, save_adversary,
        build_trader, save_trader,
        training::zero_sum::*,
    },
    env::Env,
};
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    logging,
};

fn run_experiment(save_dir: &str, eval_interval: usize) {
    let logger = logging::root(logging::stdout());
    let mut file_logger = csv::Writer::from_path(format!("{}/results.csv", save_dir)).unwrap();

    let env_builder = || Env::default_with_drift();

    // Build adversary:
    let mut trader = build_trader(env_builder().state_space(), 0.01, 0.000001);
    let mut adversary = build_adversary(env_builder().state_space(), 0.1, 0.0001);

    // Pre-train value function:
    for _ in 0..1000 {
        train_value_functions(env_builder(), &mut trader, &mut adversary);
    }

    // Run experiment:
    for i in 0.. {
        // Perform evaluation:
        if i % eval_interval == 0 {
            let r = evaluate_agents(
                env_builder,
                &mut trader,
                &mut adversary,
                i * eval_interval,
                1000,
            );

            // Serialise every agent:
            save_trader(&trader, format!("{}/trader_{}.bin", save_dir, i));
            save_adversary(&adversary, format!("{}/adversary_{}.bin", save_dir, i));

            // Log plotting data:
            info!(logger, "evaluation {}", i / eval_interval;
                "wealth" => format!("{} +/- {}", r.wealth_mean, r.wealth_stddev),
                "reward" => format!("{} +/- {}", r.reward_mean, r.reward_stddev),
                "inv" => format!("{} +/- {}", r.inv_mean, r.inv_stddev),
            );

            file_logger.serialize(r).ok();
            file_logger.flush().ok();
        }

        // Train agent for one episode:
        train_agents_once(env_builder(), &mut trader, &mut adversary);
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
        .get_matches();

    let save_dir = matches.value_of("save_dir").unwrap();
    let eval_interval: usize = matches.value_of("eval_interval").unwrap().parse().unwrap();

    run_experiment(save_dir, eval_interval);
}
