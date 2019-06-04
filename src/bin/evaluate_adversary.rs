extern crate algo_hft;
extern crate lfa;
extern crate clap;
extern crate rand;
extern crate rsrl;

use algo_hft::{
    agents::{load_trader, Trader, load_adversary, Adversary, tta},
    env::Env,
    utils::Estimate,
};
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    policies::Policy,
};

const MAX_DRIFT: f64 = 5.0;

#[derive(Debug)]
struct Record<T> {
    pub wealth: T,
    pub inv: T,
}

fn simulate_once(trader: &mut Trader, adversary: &mut Adversary) -> Record<f64> {
    let mut domain = Env::default_with_drift();

    loop {
        let d = adversary.policy.mpa(domain.emit().state());
        let a = trader.policy.mpa(domain.emit().state());

        domain.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * d - 1.0);
        let t = domain.step(tta(a));

        if t.terminated() {
            return Record {
                wealth: domain.wealth,
                inv: domain.inv_terminal,
            }
        }
    }
}

fn main() {
    let matches = App::new("AS inventory strategy simulator")
        .arg(Arg::with_name("n_simulations")
                .index(1)
                .required(true))
        .arg(Arg::with_name("trader_path")
                .index(2)
                .required(true))
        .arg(Arg::with_name("adversary_path")
                .index(3)
                .required(true))
        .get_matches();

    let n_simulations: usize = matches.value_of("n_simulations").unwrap().parse().unwrap();

    let mut trader = load_trader(matches.value_of("trader_path").unwrap().to_string());
    let mut adversary = load_adversary(matches.value_of("adversary_path").unwrap().to_string());

    let mut wealth_values: Vec<f64> = Vec::with_capacity(n_simulations);
    let mut inv_values: Vec<f64> = Vec::with_capacity(n_simulations);

    (0..n_simulations).into_iter().map(|_| simulate_once(&mut trader, &mut adversary)).for_each(|r| {
        wealth_values.push(r.wealth);
        inv_values.push(r.inv);
    });

    let summary = Record {
        wealth: Estimate::from_slice(&wealth_values),
        inv: Estimate::from_slice(&inv_values),
    };

    println!("{:#?}", summary);
}
