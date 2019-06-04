extern crate algo_hft;
extern crate lfa;
extern crate bincode;
extern crate csv;
extern crate clap;
extern crate rand;
extern crate rsrl;
#[macro_use]
extern crate serde_derive;

use algo_hft::{
    agents::{Trader, tta},
    env::Env,
    utils::Estimate,
};
use bincode::deserialize_from;
use clap::{App, Arg};
use rsrl::{
    domains::Domain,
    policies::Policy,
};
use std::{
    fs::File,
    io::{BufReader, stdout},
};

#[derive(Debug, Serialize)]
struct Record<T> {
    pub wealth: T,
    pub inv: T,
    pub average_spread: T,
}

fn simulate_once(trader: &mut Trader) -> Record<f64> {
    let mut domain = Env::default();

    let mut i = 0;
    let mut spread_sum = 0.0;

    loop {
        let a = trader.policy.mpa(domain.emit().state());
        let t = domain.step(tta(a));

        i += 1;
        spread_sum += a.1 * 2.0;

        if t.terminated() {
            return Record {
                wealth: domain.wealth,
                inv: domain.inv_terminal,
                average_spread: spread_sum / i as f64,
            }
        }
    }
}

fn main() {
    let matches = App::new("AS inventory strategy simulator")
        .arg(Arg::with_name("n_simulations")
                .index(1)
                .required(true))
        .arg(Arg::with_name("bin_path")
                .index(2)
                .required(true))
        .get_matches();

    let n_simulations: usize = matches.value_of("n_simulations").unwrap().parse().unwrap();
    let bin_path = matches.value_of("bin_path").unwrap();

    let reader = BufReader::new(File::open(bin_path).unwrap());
    let mut trader: Trader = deserialize_from(reader).unwrap();

    // let mut wealth_values: Vec<f64> = Vec::with_capacity(n_simulations);
    // let mut inv_values: Vec<f64> = Vec::with_capacity(n_simulations);
    // let mut spread_values: Vec<f64> = Vec::with_capacity(n_simulations);

    let mut csv_logger = csv::Writer::from_writer(stdout());

    (0..n_simulations).into_iter().map(|_| simulate_once(&mut trader)).for_each(|r| {
        csv_logger.serialize(r).ok();

        // wealth_values.push(r.wealth);
        // inv_values.push(r.inv);
        // spread_values.push(r.average_spread);
    });

    csv_logger.flush().ok();

    // println!("{:#?}", Record {
        // wealth: Estimate::from_slice(&wealth_values),
        // inv: Estimate::from_slice(&inv_values),
        // average_spread: Estimate::from_slice(&spread_values),
    // });
    // println!("Bull: {}", trader.policy.mpa(&vec![0.0, 5.0]).0);
    // println!("Neut: {}", trader.policy.mpa(&vec![0.0, 0.0]).0);
    // println!("Bear: {}", trader.policy.mpa(&vec![0.0, -5.0]).0);
}
